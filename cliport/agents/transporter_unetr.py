import os
import cv2
from cliport.agents.transporter import TransporterAgent
import numpy as np
import pdb

from cliport.utils import utils
import cliport.utils.visual_utils as vu
from cliport.models.streams.one_stream_conceptfusion import OneStreamAttenUnetr
from cliport.models.streams.one_stream_conceptfusion import OneStreamTransportUnetr

from cliport.models.streams.conceptfusion import conceptfusion
from cliport.models.core.attention import Attention


class UnetrAgent(TransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        atten_stream_fcn = 'Unetr'   # pick map
        key_stream_fcn = 'Unetr'     # place map
        query_stream_fcn = 'Unetr_kernel'   # crop
        self.attention = OneStreamAttenUnetr(
            stream_fcn=(atten_stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportUnetr(
            key_stream_fcn=(key_stream_fcn, None),
            query_stream_fcn=(query_stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )

    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        lang_goal = inp['lang_goal']
        # pdb.set_trace()
        out = self.attention.forward(inp_img, lang_goal, softmax=softmax)
        return out

    def attn_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0, p0_theta = frame['p0'], frame['p0_theta']
        lang_goal = frame['lang_goal']
        # pdb.set_trace()

        inp = {'inp_img': inp_img, 'lang_goal': lang_goal}
        out = self.attn_forward(inp, softmax=False)
        return self.attn_criterion(backprop, compute_err, inp, out, p0, p0_theta)

    def trans_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        p0 = inp['p0']
        lang_goal = inp['lang_goal']

        # pdb.set_trace()
        out = self.transport.forward(inp_img, p0, lang_goal, softmax=softmax)
        return out

    def transport_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0 = frame['p0']
        p1, p1_theta = frame['p1'], frame['p1_theta']
        lang_goal = frame['lang_goal']

        inp = {'inp_img': inp_img, 'p0': p0, 'lang_goal': lang_goal}
        out = self.trans_forward(inp, softmax=False)
        err, loss = self.transport_criterion(backprop, compute_err, inp, out, p0, p1, p1_theta)
        return loss, err

    
    def training_step(self, batch, batch_idx):
        self.attention.train()
        self.transport.train()
        
        frame, _ = batch  # keys of frame: ['img', 'p0', 'p0_theta', 'p1', 'p1_theta', 'perturb_params', 'lang_goal']

        # Get training losses.
        step = self.total_steps + 1
        loss0, err0 = self.attn_training_step(frame)
    
        if isinstance(self.transport, Attention):
            loss1, err1 = self.attn_training_step(frame)
        else:
            loss1, err1 = self.transport_training_step(frame)
        
        total_loss = loss0 + loss1
        self.log('tr/attn/loss', loss0)
        self.log('tr/trans/loss', loss1)
        self.log('tr/loss', total_loss)
        self.total_steps = int( self.trainer.global_step / 2)

        # self.trainer.train_loop.running_loss.append(total_loss)
        # pdb.set_trace()
        # self.check_save_iteration()

        return dict(
            loss=total_loss,
        )
    
    def validation_step(self, batch, batch_idx):
        self.attention.eval()
        self.transport.eval()

        loss0, loss1 = 0, 0
        assert self.val_repeats >= 1
        for i in range(self.val_repeats):
            frame, _ = batch
            l0, err0 = self.attn_training_step(frame, backprop=False, compute_err=True)
            loss0 += l0
            if isinstance(self.transport, Attention):
                l1, err1 = self.attn_training_step(frame, backprop=False, compute_err=True)
                loss1 += l1
            else:
                l1, err1 = self.transport_training_step(frame, backprop=False, compute_err=True)
                loss1 += l1
        loss0 /= self.val_repeats
        loss1 /= self.val_repeats
        val_total_loss = loss0 + loss1

        self.val_output_list.append(
            dict(
                val_loss=val_total_loss,
                val_loss0=loss0,
                val_loss1=loss1,
                val_attn_dist_err=err0['dist'],
                val_attn_theta_err=err0['theta'],
                val_trans_dist_err=err1['dist'],
                val_trans_theta_err=err1['theta'],
            )
        )
    
    def test_step(self, batch, batch_idx):
        self.attention.eval()
        self.transport.eval()

        loss0, loss1 = 0, 0
        assert self.val_repeats >= 1
        # pdb.set_trace()
        for i in range(self.val_repeats):
            frame, _ = batch
            l0, err0, out_attn = self.attn_training_step(frame, backprop=False, compute_err=True, return_output=True)
            loss0 += l0
            if isinstance(self.transport, Attention):
                l1, err1, out_attn = self.attn_training_step(frame, backprop=False, compute_err=True, return_output=True)
                loss1 += l1
            else:
                l1, err1, out_trans = self.transport_training_step(frame, backprop=False, compute_err=True, return_output=True)
                loss1 += l1
        loss0 /= self.val_repeats
        loss1 /= self.val_repeats
        val_total_loss = loss0 + loss1        

        #import pdb; pdb.set_trace()
        img = frame['img'][:,:,:3]
        pick_place = frame['p0']
        place_place = frame['p1']
        pick_radius = frame['pick_radius']
        place_radius = frame['place_radius']
        text = frame['lang_goal']

        img = img.astype(np.uint8)

        #save heatmap images
        out_attn = out_attn.reshape(320,160).detach().cpu().numpy()
        out_trans = out_trans.detach().cpu().numpy()
        save_path = os.path.join(self.cfg['train']['train_dir'], 'real_vis')
        os.makedirs(save_path, exist_ok=True)
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        name = self.name.split('-')[0]
        save = vu.save_tensor_with_heatmap(image, out_attn,
            f'{save_path}/{name}_pick{batch_idx + 1:06d}.png',
            l=text)
        save = vu.save_tensor_with_heatmap(image, out_trans,
            f'{save_path}/{name}_place{batch_idx + 1:06d}.png',
            l=text)
        # save gt images
        # brg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.circle(brg, (pick_place[1], pick_place[0]), int(pick_radius), (0, 255, 0), 2)
        # cv2.circle(brg, (place_place[1], place_place[0]), int(place_radius), (0, 0, 255), 2)
        # cv2.putText(brg, text, (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
        # foler = 'data_gt_real_images'
        # idx = len(os.listdir(foler))
        # cv2.imwrite(f'data_gt_real_images/real{idx}.png', brg)
        

        # whether successful pick and place ?
        if err0['dist'] < frame['pick_radius']:
            success_pick = 1
        else:
            success_pick = 0
        
        if err1['dist'] < frame['place_radius']:
            success_place = 1
        else:
            success_place = 0
        
        if err0['dist'] < frame['pick_radius'] and err1['dist'] < frame['place_radius']:
            success = 1
        else:
            success = 0
                
        self.test_output_list.append( 
                dict(
                val_loss=val_total_loss,
                val_loss0=loss0,
                val_loss1=loss1,
                val_attn_dist_err=err0['dist'],
                val_attn_theta_err=err0['theta'],
                val_trans_dist_err=err1['dist'],
                val_trans_theta_err=err1['theta'],
                success=success,
                success_pick=success_pick,
                success_place=success_place
            )
        )

    def act(self, obs, info, goal=None):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        # Get heightmap from RGB-D images.
        # pdb.set_trace()
        img = self.test_ds.get_image(obs)
        lang_goal = info['lang_goal']

        frame = {'img': img, 'lang_goal': lang_goal}

        # Attention model forward pass.
        pick_inp = {'inp_img': img, 'lang_goal': lang_goal}
        pick_conf = self.attn_forward(pick_inp)
        pick_conf = pick_conf.detach().cpu().numpy()
        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0_pix = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

        # Transport model forward pass.
        place_inp = {'inp_img': img, 'p0': p0_pix, 'lang_goal': lang_goal}
        place_conf = self.trans_forward(place_inp)
        place_conf = place_conf.permute(1, 2, 0)
        place_conf = place_conf.detach().cpu().numpy()
        argmax = np.argmax(place_conf)
        argmax = np.unravel_index(argmax, shape=place_conf.shape)
        p1_pix = argmax[:2]
        p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

        # Pixels to end effector poses.
        hmap = img[:, :, 3]
        p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
        p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
        p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
        p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))

        return {
            'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
            'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw)),
            'pick': [p0_pix[0], p0_pix[1], p0_theta],
            'place': [p1_pix[0], p1_pix[1], p1_theta],
        }