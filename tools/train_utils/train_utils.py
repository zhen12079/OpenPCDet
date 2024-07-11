import glob
import os

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        loss, tb_dict, disp_dict = model_func(model, batch)

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)













import copy
import glob
import os

import torch
import tqdm
import time
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils
from torch.cuda import amp

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        # logger.info(str(m)+" is set eval")
def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False,scaler=None,
                    cur_epoch=None, total_epochs=None,
                    use_amp=False,logger=None,cfg=None):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)
    start_it = accumulated_iter % total_it_each_epoch
    # error_flag=0
    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()
        losses_m = common_utils.AverageMeter()
    model.train()

    if isinstance(cfg.MODEL.NOT_TRAIN_IN_MULTI_TASK, list) and cfg.MODEL.NOT_TRAIN_IN_MULTI_TASK != []:
        if "vfe" in cfg.MODEL.NOT_TRAIN_IN_MULTI_TASK:
            model.module.vfe.apply(set_bn_eval)
        if "backbone_2d.blocks" in cfg.MODEL.NOT_TRAIN_IN_MULTI_TASK:
            model.module.backbone_2d.blocks.apply(set_bn_eval)
        if "backbone_2d.deblocks" in cfg.MODEL.NOT_TRAIN_IN_MULTI_TASK:
            model.module.backbone_2d.deblocks.apply(set_bn_eval)
        if "dense_head" in cfg.MODEL.NOT_TRAIN_IN_MULTI_TASK:
            model.module.dense_head.apply(set_bn_eval)
            
        if "backbone_2d.segblocks" in cfg.MODEL.NOT_TRAIN_IN_MULTI_TASK:
            model.module.backbone_2d.segblocks.apply(set_bn_eval)
        if "backbone_2d.seg_out_conv" in cfg.MODEL.NOT_TRAIN_IN_MULTI_TASK:
            model.module.backbone_2d.seg_out_conv.apply(set_bn_eval)
        if "segment_head" in cfg.MODEL.NOT_TRAIN_IN_MULTI_TASK:
            model.module.segment_head.apply(set_bn_eval)

        if "backbone_2d.laneblocks" in cfg.MODEL.NOT_TRAIN_IN_MULTI_TASK:
            model.module.backbone_2d.laneblocks.apply(set_bn_eval)
        if "backbone_2d.mixsegnet" in cfg.MODEL.NOT_TRAIN_IN_MULTI_TASK:
            model.module.backbone_2d.mixsegnet.apply(set_bn_eval)
        if "lane_head" in cfg.MODEL.NOT_TRAIN_IN_MULTI_TASK:
            model.module.lane_head.apply(set_bn_eval)

    # model.module.vfe.apply(set_bn_eval)
    # model.module.backbone_2d.blocks.apply(set_bn_eval)
    # model.module.backbone_2d.deblocks.apply(set_bn_eval)
    # model.module.dense_head.apply(set_bn_eval)

    # model.module.backbone_2d.segblocks.apply(set_bn_eval)
    # model.module.backbone_2d.seg_out_conv.apply(set_bn_eval)
    # model.module.segment_head.apply(set_bn_eval)

    # model.module.backbone_2d.laneblocks.apply(set_bn_eval)
    # model.module.backbone_2d.mixsegnet.apply(set_bn_eval)
    # model.module.lane_head.apply(set_bn_eval)

    for cur_it in range(total_it_each_epoch):
        end = time.time()
        # lr_scheduler.step(accumulated_iter)
        while True:
            try:
                batch = next(dataloader_iter)
                assert batch!=None
                break
            except Exception as e:
                logger.info("problem is:"+str(e))
                print(str(e))
                if isinstance(e,StopIteration):
                    dataloader_iter = iter(train_loader)
                    logger.info('new iters')
                continue
        data_timer = time.time()
        cur_data_time = data_timer - end
        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']
        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
            tb_log.add_scalar('meta_data/epoch', int((accumulated_iter+total_it_each_epoch)/total_it_each_epoch), accumulated_iter)

        optimizer.zero_grad()
        if use_amp:
            with amp.autocast(enabled=True):
                loss, tb_dict, disp_dict = model_func(model, batch)
        else:
            loss, tb_dict, disp_dict = model_func(model, batch)

        forward_timer = time.time()
        cur_forward_time = forward_timer - data_timer
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        lr_scheduler.step(accumulated_iter)
        accumulated_iter += 1

        cur_batch_time = time.time() - end
        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)
        # if error_flag:
        #     # error_flag=0
        #     print("go on new batch3")
        #     print(rank)
        # log to console and tensorboard
        if rank == 0:
            batch_size = batch.get('batch_size', None)

            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            losses_m.update(loss.item() , batch_size)

            disp_dict.update({
                'loss': loss.item(), 'lr': cur_lr, 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})', 'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
            })

            use_logger_to_record = True
            logger_iter_interval = 50
            if use_logger_to_record:
                # import pdb; pdb.set_trace()
                if accumulated_iter % logger_iter_interval == 0 or cur_it == start_it or cur_it + 1 == total_it_each_epoch:
                    trained_time_past_all = tbar.format_dict['elapsed']
                    second_each_iter = pbar.format_dict['elapsed'] / max(cur_it - start_it + 1, 1.0)

                    trained_time_each_epoch = pbar.format_dict['elapsed']
                    remaining_second_each_epoch = second_each_iter * (total_it_each_epoch - cur_it)
                    remaining_second_all = second_each_iter * ((total_epochs - cur_epoch) * total_it_each_epoch - cur_it)
                    
                    logger.info(
                        'Train: {:>4d}/{} ({:>3.0f}%) [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                        'LR: {lr:.3e}  '
                        f'Time cost: {tbar.format_interval(trained_time_each_epoch)}/{tbar.format_interval(remaining_second_each_epoch)} ' 
                        f'[{tbar.format_interval(trained_time_past_all)}/{tbar.format_interval(remaining_second_all)}]  '
                        'Acc_iter {acc_iter:<10d}  '
                        'Data time: {data_time.val:.2f}({data_time.avg:.2f})  '
                        'Forward time: {forward_time.val:.2f}({forward_time.avg:.2f})  '
                        'Batch time: {batch_time.val:.2f}({batch_time.avg:.2f})'.format(
                            cur_epoch+1,total_epochs, 100. * (cur_epoch+1) / total_epochs,
                            cur_it,total_it_each_epoch, 100. * cur_it / total_it_each_epoch,
                            loss=losses_m,
                            lr=cur_lr,
                            acc_iter=accumulated_iter,
                            data_time=data_time,
                            forward_time=forward_time,
                            batch_time=batch_time
                            )
                    )
                    
                    # if show_gpu_stat and accumulated_iter % (3 * logger_iter_interval) == 0:
                    #     # To show the GPU utilization, please install gpustat through "pip install gpustat"
                    #     gpu_info = os.popen('gpustat').read()
                    #     logger.info(gpu_info)
            else:                
                pbar.update()
                pbar.set_postfix(dict(total_it=accumulated_iter))
                tbar.set_postfix(disp_dict)

            if tb_log is not None:
                try:
                    tb_log.add_scalar('train/loss', loss, accumulated_iter)
                    tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                    for key, val in tb_dict.items():
                        tb_log.add_scalar('train/' + key, val, accumulated_iter)
                except Exception as e:
                    print(str(e))
        # if rank == 0:
        #     data_time.update(avg_data_time)
        #     forward_time.update(avg_forward_time)
        #     batch_time.update(avg_batch_time)

        #     disp_dict.update({
        #         'loss': loss.item(), 'lr': cur_lr,
        #         'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
        #         'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})',
        #         'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
        #     })
        #     pbar.update()
        #     pbar.set_postfix(dict(total_it=accumulated_iter))
        #     tbar.set_postfix(disp_dict)
        #     tbar.refresh()
        #     if tb_log is not None:
        #         # tb_log.add_scalar('train/loss', loss, accumulated_iter)
        #         tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
        #         for key, val in tb_dict.items():
        #             tb_log.add_scalar('train/' + key, val, accumulated_iter)

    if rank == 0:
        pbar.close()
    return accumulated_iter,disp_dict


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False,logger=None,use_amp=False,args=None,cfg=None):
    accumulated_iter = start_iter
    if use_amp:
        logger.info("use amp")
        scaler = amp.GradScaler(enabled=True)
    else:
        scaler = None
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            #try:
            accumulated_iter,disp_dict = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                cur_epoch=cur_epoch, total_epochs=total_epochs,
                scaler=scaler,
                use_amp=use_amp,
                logger=logger,
                cfg=cfg
            )
            #except Exception as e:
            #    logger.info(str(e))
            #    continue
            # save trained model
            trained_epoch = cur_epoch + 1
            if logger:
                logger.info("trained_epoch: "+str(trained_epoch) + "/" + str(total_epochs)+" "+str(disp_dict))
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)

                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )
                # if args.quant_qat:
                #     try:
                #         save_path = str(ckpt_name)+"_QAT.onnx"
                #         try:
                #             export_onnx(model.module, save_path)
                #         except:
                #             export_onnx(model, save_path)
                #     except Exception as e:
                #         print(str(e))
                #         continue



def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            # compute_amax(model.module)
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            # compute_amax(model)
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)

