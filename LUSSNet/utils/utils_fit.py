import os
import torch
import torch.nn.functional as F
import torch
from nets.LUSSNet_training import CE_Loss, Dice_loss, Focal_Loss
from tqdm import tqdm
from utils.utils import get_lr
from utils.utils_metrics import f_score


def WDD_loss(inputs, target, beta=1, smooth=1e-5, ε=1e-6):
    n, c, h, w = inputs.size()
    if target.ndim == 4:  # one-hot: (B, H, W, C)
        if target.shape[-1] == c + 1:
            target = target[..., :-1]
        if target.shape[-1] != c:
            raise ValueError("class+1")
        target = target.permute(0, 3, 1, 2).float()  # -> (B, C, H, W)
    elif target.ndim == 3:  # index: (B, H, W)
        target = F.one_hot(target.long(), num_classes=c).permute(0, 3, 1, 2).float()
    else:
        raise ValueError("NO target SHAPE")

    if inputs.shape[2:] != target.shape[2:]:
        inputs = F.interpolate(inputs, size=target.shape[2:], mode="bilinear", align_corners=True)
    probs = torch.softmax(inputs, dim=1)  # [B, C, H, W]
    probs_flat = probs.permute(0, 2, 3, 1).reshape(n, -1, c)   # [B, H*W, C]
    target_flat = target.permute(0, 2, 3, 1).reshape(n, -1, c) # [B, H*W, C]
    tp = torch.sum(target_flat * probs_flat, dim=[0, 1])       # [C]
    fp = torch.sum(probs_flat, dim=[0, 1]) - tp
    fn = torch.sum(target_flat, dim=[0, 1]) - tp
    dice_score = ((1 + beta**2) * tp + smooth) / ((1 + beta**2) * tp + beta**2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(dice_score)
    mean_pred = torch.mean(probs_flat, dim=[0, 1])     # (C,)
    mean_tgt  = torch.mean(target_flat, dim=[0, 1])    # (C,)
    var_pred  = torch.var(probs_flat, dim=[0, 1])      # (C,)
    var_tgt   = torch.var(target_flat, dim=[0, 1])     # (C,)
    nwd_loss = torch.mean((mean_pred - mean_tgt) ** 2 +
                     (torch.sqrt(var_pred + smooth) - torch.sqrt(var_tgt + smooth)) ** 2)
    ratio = dice_loss / (nwd_loss + ε)
    α = ratio / (1 + ratio)
    Total_Loss = α * dice_loss + (1 - α) * nwd_loss
    return Total_Loss



def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, cuda, wdd_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period,
                  save_dir, log_dir, local_rank=0, eval_period=5):
    total_loss = 0
    total_f_score = 0

    val_loss = 0
    val_f_score = 0

    if local_rank == 0:
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:

            outputs = model_train(imgs)

            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if wdd_loss:

                main_dice = WDD_loss(outputs, labels)
                loss = loss + main_dice

            with torch.no_grad():

                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():

                outputs = model_train(imgs)

                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if wdd_loss:
                    main_dice = WDD_loss(outputs, labels)
                    loss = loss + main_dice

                with torch.no_grad():

                    _f_score = f_score(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()

        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            outputs = model_train(imgs)

            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if wdd_loss:

                main_dice = WDD_loss(outputs, labels)
                loss = loss + main_dice

            _f_score = f_score(outputs, labels)

            val_loss += loss.item()
            val_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                'f_score': val_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()

        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))


        epoch_1 = epoch + 1

        if (epoch_1 % eval_period) == 0:

            with open(os.path.join(log_dir, "epoch_miou.txt"), 'r') as file:
                lines = file.readlines()

                last_line = lines[-1].strip() if lines else None
                l = len(lines)
                last_l = l - 1
                if len(lines) > 2:
                    sub_list = [item.strip() for item in lines] if lines else None
                    max_value = max(sub_list[:last_l])
                    if float(last_line) > float(max_value):
                        print('Save best model to best_epoch_weights.pth')

                        torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))


def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda, dice_loss,
                         focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss = 0
    total_f_score = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:

            outputs = model_train(imgs)

            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:

                main_dice = WDD_loss(outputs, labels)
                loss = loss + main_dice


            with torch.no_grad():

                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():

                outputs = model_train(imgs)

                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = WDD_loss(outputs, labels)
                    loss = loss + main_dice

                with torch.no_grad():

                    _f_score = f_score(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, total_loss / epoch_step)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f' % (total_loss / epoch_step))

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'ep%03d-loss%.3f.pth' % ((epoch + 1), total_loss / epoch_step)))

        if len(loss_history.losses) <= 1 or (total_loss / epoch_step) <= min(loss_history.losses):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))