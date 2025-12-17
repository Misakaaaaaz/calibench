"""The CarliniWagnerL2 attack."""
import torch


INF = float("inf")


def carlini_wagner_l2(
    model_fn,
    x,
    n_classes,
    y=None,
    targeted=False,
    lr=5e-3,
    confidence=0,
    clip_min=0,
    clip_max=1,
    initial_const=1e-2,
    binary_search_steps=5,
    max_iterations=1000,
    debug=True,
    early_stop=False,
):
    """
    This attack was originally proposed by Carlini and Wagner. It is an
    iterative attack that finds adversarial examples on many defenses that
    are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644

    At a high level, this attack is an iterative attack using Adam and
    a specially-chosen loss function to find adversarial examples with
    lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.

    :param model_fn: a callable that takes an input tensor and returns
              the model logits. The logits should be a tensor of shape
              (n_examples, n_classes).
    :param x: input tensor of shape (n_examples, ...), where ... can
              be any arbitrary dimension that is compatible with
              model_fn.
    :param n_classes: the number of classes.
    :param y: (optional) Tensor with true labels. If targeted is true,
              then provide the target label. Otherwise, only provide
              this parameter if you'd like to use true labels when
              crafting adversarial samples. Otherwise, model predictions
              are used as labels to avoid the "label leaking" effect
              (explained in this paper:
              https://arxiv.org/abs/1611.01236). If provide y, it
              should be a 1D tensor of shape (n_examples, ).
              Default is None.
    :param targeted: (optional) bool. Is the attack targeted or
              untargeted? Untargeted, the default, will try to make the
              label incorrect. Targeted will instead try to move in the
              direction of being more like y.
    :param lr: (optional) float. The learning rate for the attack
              algorithm. Default is 5e-3.
    :param confidence: (optional) float. Confidence of adversarial
              examples: higher produces examples with larger l2
              distortion, but more strongly classified as adversarial.
              Default is 0.
    :param clip_min: (optional) float. Minimum float value for
              adversarial example components. Default is 0.
    :param clip_max: (optional) float. Maximum float value for
              adversarial example components. Default is 1.
    :param initial_const: The initial tradeoff-constant to use to tune the
              relative importance of size of the perturbation and
              confidence of classification. If binary_search_steps is
              large, the initial constant is not important. A smaller
              value of this constant gives lower distortion results.
              Default is 1e-2.
    :param binary_search_steps: (optional) int. The number of times we
              perform binary search to find the optimal tradeoff-constant
              between norm of the perturbation and confidence of the
              classification. Default is 5.
    :param max_iterations: (optional) int. The maximum number of
              iterations. Setting this to a larger value will produce
              lower distortion results. Using only a few iterations
              requires a larger learning rate, and will produce larger
              distortion results. Default is 1000.
    :param debug: (optional) bool. Whether to print debug information.
              Default is True.
    :param early_stop: (optional) bool. Whether to stop as soon as a
              successful adversarial example is found. This might not give
              the lowest L2 norm but can be faster. Default is False.
              
    :return: A tuple of (adversarial_examples, l2_norms, success_mask):
            - adversarial_examples: tensor of adversarial examples
            - l2_norms: tensor with L2 norms of perturbations
            - success_mask: boolean tensor indicating which samples were successfully attacked
    """

    def compare(pred, label, is_logits=False):
        """
        A helper function to compare prediction against a label.
        Returns true if the attack is considered successful.

        :param pred: can be either a 1D tensor of logits or a predicted
                class (int).
        :param label: int. A label to compare against.
        :param is_logits: (optional) bool. If True, treat pred as an
                array of logits. Default is False.
        """

        # Convert logits to predicted class if necessary
        if is_logits:
            pred_copy = pred.clone().detach()
            pred_copy[label] += -confidence if targeted else confidence
            pred = torch.argmax(pred_copy)

        return pred == label if targeted else pred != label

    if debug:
        print(f"CW Attack Parameters:")
        print(f"- targeted: {targeted}")
        print(f"- lr: {lr}")
        print(f"- confidence: {confidence}")
        print(f"- initial_const: {initial_const}")
        print(f"- binary_search_steps: {binary_search_steps}")
        print(f"- max_iterations: {max_iterations}")
        print(f"- clip range: [{clip_min}, {clip_max}]")
        print(f"- early_stop: {early_stop}")

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        with torch.no_grad():
            pred = model_fn(x)
        y = torch.argmax(pred, 1)
        
        if debug:
            probs = torch.nn.functional.softmax(pred, dim=1)
            max_prob, _ = torch.max(probs, dim=1)
            print(f"Using model predictions as labels: {y.cpu().numpy()}")
            print(f"Prediction confidence: {max_prob.cpu().numpy()}")

    # Initialize some values needed for binary search on const
    lower_bound = [0.0] * len(x)
    upper_bound = [1e10] * len(x)
    const = x.new_ones(len(x), 1) * initial_const

    o_bestl2 = [INF] * len(x)
    o_bestscore = [-1.0] * len(x)
    x = torch.clamp(x, clip_min, clip_max)
    ox = x.clone().detach()  # save the original x
    o_bestattack = x.clone().detach()

    # 添加一个标志，指示是否已找到所有样本的成功攻击
    all_successful = False

    if debug:
        print(f"Input shape: {x.shape}")
        print(f"Input min: {x.min().item()}, max: {x.max().item()}")

    # Map images into the tanh-space
    x = (x - clip_min) / (clip_max - clip_min)
    x = torch.clamp(x, 0, 1)
    x = x * 2 - 1
    x = torch.arctanh(x * 0.999999)
    
    if debug:
        print(f"After tanh transform - min: {x.min().item()}, max: {x.max().item()}")

    # Prepare some variables
    modifier = torch.zeros_like(x, requires_grad=True)
    y_onehot = torch.nn.functional.one_hot(y, n_classes).to(torch.float)

    # Define loss functions and optimizer
    f_fn = lambda real, other, targeted: torch.max(
        ((other - real) if targeted else (real - other)) + confidence,
        torch.tensor(0.0).to(real.device),
    )
    l2dist_fn = lambda x, y: torch.pow(x - y, 2).sum(list(range(len(x.size())))[1:])
    optimizer = torch.optim.Adam([modifier], lr=lr)

    # Outer loop performing binary search on const
    for outer_step in range(binary_search_steps):
        if debug:
            print(f"\n--- Binary search step {outer_step+1}/{binary_search_steps} ---")
            print(f"Current const range: min={min(lower_bound):.4e}, max={min([ub for ub in upper_bound if ub < 1e10] or [1e10]):.4e}")

        # Initialize some values needed for the inner loop
        bestl2 = [INF] * len(x)
        bestscore = [-1.0] * len(x)

        if debug:
            success_count = sum(1 for score in o_bestscore if score != -1.0)
            print(f"Current best success rate: {success_count}/{len(x)} ({success_count/len(x)*100:.2f}%)")
            if success_count > 0:
                avg_l2 = sum(l2 for l2 in o_bestl2 if l2 != INF) / success_count
                print(f"Average L2 for successful attacks: {avg_l2:.4f}")
            
            # 如果开启early_stop且所有样本都已成功，则提前结束
            if early_stop and success_count == len(x):
                print("All samples successfully attacked. Stopping early.")
                all_successful = True
                break

        if all_successful:
            break

        prev_loss = float('inf')
        
        # 跟踪本次binary search中是否有新的成功攻击
        new_success_in_iteration = False
        
        for i in range(max_iterations):
            # One attack step
            new_x = (torch.tanh(modifier + x) + 1) / 2
            new_x = new_x * (clip_max - clip_min) + clip_min
            
            logits = model_fn(new_x)
            
            if debug and (i == 0 or (i+1) % 100 == 0 or i == max_iterations-1):
                with torch.no_grad():
                    predicted = torch.argmax(logits, dim=1)
                    success_mask = predicted != y if not targeted else predicted == y
                    current_success_rate = success_mask.float().mean().item() * 100
                    
                    l2_norms = l2dist_fn(new_x, ox)
                    avg_l2 = l2_norms.mean().item()
                    
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    max_probs, _ = torch.max(probs, dim=1)
                    avg_conf = max_probs.mean().item()
                    
                    print(f"Iteration {i+1}/{max_iterations}:")
                    print(f"  Current success rate: {current_success_rate:.2f}%")
                    print(f"  Average L2 norm: {avg_l2:.4f}")
                    print(f"  Average confidence: {avg_conf:.4f}")

            real = torch.sum(y_onehot * logits, 1)
            other, _ = torch.max((1 - y_onehot) * logits - y_onehot * 1e4, 1)

            optimizer.zero_grad()
            f = f_fn(real, other, targeted)
            l2 = l2dist_fn(new_x, ox)
            loss = torch.mean(const * f + l2)
            
            if debug and (i == 0 or (i+1) % 100 == 0 or i == max_iterations-1):
                loss_diff = abs(prev_loss - loss.item())
                print(f"  Loss: {loss.item():.4f} (change: {loss_diff:.4e})")
                prev_loss = loss.item()
            
            loss.backward()
            optimizer.step()

            # Update best results
            # 添加一个标志，记录此次迭代是否有任何样本攻击成功
            iter_success = False
            
            for n, (l2_n, logits_n, new_x_n) in enumerate(zip(l2, logits, new_x)):
                y_n = y[n]
                succeeded = compare(logits_n, y_n, is_logits=True)
                
                if succeeded:
                    iter_success = True
                
                if l2_n < o_bestl2[n] and succeeded:
                    pred_n = torch.argmax(logits_n)
                    o_bestl2[n] = l2_n
                    o_bestscore[n] = pred_n
                    o_bestattack[n] = new_x_n
                    # l2_n < o_bestl2[n] implies l2_n < bestl2[n] so we modify inner loop variables too
                    bestl2[n] = l2_n
                    bestscore[n] = pred_n
                    
                    new_success_in_iteration = True
                    
                    if debug and i > 0 and i % 100 == 0:
                        print(f"  Found better attack for sample {n}: L2={l2_n.item():.4f}, pred={pred_n.item()}")
                        
                elif l2_n < bestl2[n] and succeeded:
                    bestl2[n] = l2_n
                    bestscore[n] = torch.argmax(logits_n)
            
            # 如果开启了early_stop且当前迭代中找到了成功攻击，检查是否所有样本都攻击成功
            if early_stop and iter_success:
                all_samples_success = all(score != -1.0 for score in o_bestscore)
                if all_samples_success:
                    if debug:
                        print(f"  All samples successfully attacked at iteration {i+1}. Stopping inner loop early.")
                    all_successful = True
                    break

        if all_successful:
            if debug:
                print("All samples successfully attacked. Stopping binary search early.")
            break

        # Binary search step
        for n in range(len(x)):
            y_n = y[n]
            
            if debug:
                status = "Success" if compare(bestscore[n], y_n) and bestscore[n] != -1 else "Failure"
                print(f"Sample {n}: {status}, L2={bestl2[n] if bestl2[n] != INF else 'INF'}")

            if compare(bestscore[n], y_n) and bestscore[n] != -1:
                # Success, divide const by two
                upper_bound[n] = min(upper_bound[n], const[n].item())
                if upper_bound[n] < 1e9:
                    const[n] = (lower_bound[n] + upper_bound[n]) / 2
                    
                    if debug:
                        print(f"  Success - decreasing const to {const[n].item():.4e} for sample {n}")
            else:
                # Failure, either multiply by 10 if no solution found yet
                # or do binary search with the known upper bound
                lower_bound[n] = max(lower_bound[n], const[n].item())
                if upper_bound[n] < 1e9:
                    const[n] = (lower_bound[n] + upper_bound[n]) / 2
                    
                    if debug:
                        print(f"  Failure - increasing const to {const[n].item():.4e} for sample {n}")
                else:
                    const[n] *= 10
                    
                    if debug:
                        print(f"  Failure - multiplying const by 10 to {const[n].item():.4e} for sample {n}")

    if debug:
        success_count = sum(1 for score in o_bestscore if score != -1.0)
        print(f"\nFinal results:")
        print(f"Success rate: {success_count}/{len(x)} ({success_count/len(x)*100:.2f}%)")
        
        if success_count > 0:
            avg_l2 = sum(l2 for l2 in o_bestl2 if l2 != INF) / success_count
            print(f"Average L2 for successful attacks: {avg_l2:.4f}")
        
        print("\nPer-sample results:")
        for n in range(len(x)):
            status = "Success" if o_bestscore[n] != -1.0 else "Failure"
            l2_val = o_bestl2[n] if o_bestl2[n] != INF else "INF"
            print(f"Sample {n}: {status}, L2={l2_val}")

    # 创建成功攻击的掩码
    success_mask = torch.tensor([score != -1.0 for score in o_bestscore], device=x.device)
    
    # 获取最终的L2范数
    l2_norms = torch.tensor([l2 if l2 != INF else 0.0 for l2 in o_bestl2], device=x.device)
    
    # 返回对抗样本、L2范数和成功掩码
    return o_bestattack.detach(), l2_norms, success_mask


if __name__ == "__main__":
    x = torch.clamp(torch.randn(5, 10), 0, 1)
    y = torch.randint(0, 9, (5,))
    model_fn = lambda x: x

    # targeted
    new_x, l2_norms, success_mask = carlini_wagner_l2(model_fn, x, 10, targeted=True, y=y)
    new_pred = model_fn(new_x)
    new_pred = torch.argmax(new_pred, 1)

    # untargeted
    new_x_untargeted, l2_norms_untargeted, success_mask_untargeted = carlini_wagner_l2(model_fn, x, 10, targeted=False, y=y)
    new_pred_untargeted = model_fn(new_x_untargeted)
    new_pred_untargeted = torch.argmax(new_pred_untargeted, 1)
