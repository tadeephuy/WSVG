import os
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from scipy import ndimage
from health_multimodal.image.data.io import load_image
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
from health_multimodal.image.model.pretrained import get_biovil_image_encoder
from text_model import TextEncoder, get_cxr_bert
from utils import convert_similarity_to_image_size


class SelfEQ(nn.Module):
    def __init__(self, w_con=1, w_sim=1, w_cst=1, k=0.8):
        super().__init__()
        self.image_model = self.create_image_model()
        self.text_model = self.create_text_model()
        
        # loss
        self.contrastive_loss = SoftCLIPLoss()
        self.similarity_loss = nn.SmoothL1Loss(reduction='none')
        
        # hyperparams
        self.k = k
        self.w_con = w_con
        self.w_sim = w_sim
        self.w_cst = w_cst

    def create_image_model(self):
        image_model = get_biovil_image_encoder("resnet50")
        image_model.load_state_dict(
            torch.load('/home/huyta/Githubs/XMedCLIP/biovil_image_resnet50_proj_size_128.pt')
        )
        return image_model

    def create_text_model(self):
        tokenizer, text_model = get_cxr_bert()
        text_model = TextEncoder(tokenizer=tokenizer, text_model=text_model)
        return text_model
    
    def _get_gradcam(self, output, local_embeddings, return_shape=False):
        # print(output.shape)
        grad_wrt_f = torch.autograd.grad(
            outputs=output, inputs=local_embeddings, 
            grad_outputs=torch.ones_like(output), create_graph=True)[0]
        
        # grad_wrt_f = grad_wrt_f.clamp(0)
        grad_wrt_f = grad_wrt_f.mean((2,3), keepdim=True)

        gradcam = (local_embeddings*grad_wrt_f).mean(1)
        B, H, W = gradcam.shape
        gradcam = gradcam.clamp(0).view(B, -1)
        gradcam -= gradcam.min(dim=1, keepdim=True)[0]
        gradcam /= (gradcam.max(dim=1, keepdim=True)[0]+1e-7)
        if return_shape:
            return gradcam, H, W
        return gradcam
        

    def forward(self, img_tensor, tokens, img_label, txt_label):
        text_emb = self.text_model.get_embeddings_from_tokens(tokens, normalize=True)
        
        img_emb = self.image_model.forward(img_tensor)
        projected_global_embedding = F.normalize(img_emb.projected_global_embedding, dim=1)
        projected_patch_embeddings = img_emb.projected_patch_embeddings

        output = torch.mm(text_emb,projected_global_embedding.t())
        
        # contrastive loss
        loss_con = self.contrastive_loss(output, img_label, txt_label)
        
        # similarity loss
        gradcam = self._get_gradcam(torch.diagonal(output), projected_patch_embeddings)
        gradcam_org, gradcam_syn = gradcam.chunk(2)
        loss_sim = self.similarity_loss(gradcam_org, gradcam_syn).mean(-1).mean()
        
        # consistency loss
        if self.w_cst == 0:
            loss_cst = torch.tensor(0.0, device=loss_sim.device)
        else:
            with torch.no_grad():
                sum_gradcam = gradcam_org + gradcam_syn
                mask_gradcam = sum_gradcam >= self.k
            roi_org = gradcam_org*mask_gradcam
            roi_syn = gradcam_syn*mask_gradcam
            
            m_roi_org = roi_org.sum(dim=-1)/(mask_gradcam.sum(dim=-1)+1e-7)
            s_roi_org = roi_org.std(dim=-1)
            
            m_roi_syn = roi_syn.sum(dim=-1)/(mask_gradcam.sum(dim=-1)+1e-7)
            s_roi_syn = roi_syn.std(dim=-1)
            
            loss_cst = (s_roi_org + s_roi_syn 
                        + torch.relu(self.k/2 - m_roi_org)
                        + torch.relu(self.k/2 - m_roi_syn)).mean()
        
        loss = self.w_con*loss_con + self.w_sim*loss_sim + self.w_cst*loss_cst
        info = {
            'loss': loss.item(),
            'loss_con': loss_con.item(),
            'loss_sim': loss_sim.item(),
            'loss_cst': loss_cst.item()
        }
        return loss, info

    def save(self, p, overwrite=False):
        os.makedirs(p, exist_ok=overwrite)
        torch.save(self.image_model.state_dict(), 
                   os.path.join(p,f'img_model.pt'))
        torch.save(self.text_model.model.state_dict(), 
                   os.path.join(p,f'txt_model.pt'))

    def load(self, img_p, txt_p):
        img_weight = torch.load(img_p, weights_only=True)
        txt_weight = torch.load(txt_p, weights_only=True)
        self.image_model.load_state_dict(img_weight)
        self.text_model.model.load_state_dict(txt_weight)

    def inference_vg(self, prompt, img_path, 
                     size=512, device='cpu',
                     viz=False, ax=None):
        # with torch.set_grad_enabled(True):
        text_emb = self.text_model.get_embeddings_from_prompt(
            prompts=prompt,
            normalize=True,
            verbose=False,
        )
        transform = create_chest_xray_transform_for_inference(size,size)
        image = load_image(Path(img_path))
        x = transform(image).unsqueeze(0).to(device)
        y = self.image_model.forward(x)

        img_emb = y.projected_global_embedding
        img_emb = F.normalize(img_emb, dim=1)

        output = img_emb@text_emb.T

        cam, H, W = self._get_gradcam(output=output, 
                                local_embeddings=y.projected_patch_embeddings,
                                return_shape=True)
        cam = cam.view(H, W)
        cam = cam.detach().cpu()

        width, height = image.size
        
        sim_map = convert_similarity_to_image_size(cam, width, height, size,size)
        

        if viz: 
            if ax is not None:
                ax.imshow(image, cmap='gray')
                ax.imshow(sim_map, alpha=0.6)
            else:
                plt.imshow(image, cmap='gray')
                plt.imshow(sim_map, alpha=0.4, cmap='jet')
                # os.makedirs(os.path.dirname(viz), exist_ok=True)
                # plt.savefig(viz, bbox_inches='tight', dpi=100)
                # plt.close()
        return sim_map
    
    @torch.no_grad()
    def inference_vg_simmap(self, prompt, img_path, 
                     size=512, device='cpu',
                     viz=False, ax=None):
        text_emb = self.text_model.get_embeddings_from_prompt(
            prompts=prompt,
            normalize=True,
            verbose=False,
        )
        transform = create_chest_xray_transform_for_inference(size,size)
        image = load_image(Path(img_path))
        x = transform(image).unsqueeze(0).to(device)
        y = self.image_model.forward(x)

        local_embeddings = F.normalize(y.projected_patch_embeddings, dim=1)
        local_embeddings = local_embeddings.permute(0,2,3,1)[0]
        
        sim_map = self._get_similarity_map_from_embeddings(local_embeddings, text_emb)
        sim_map = (sim_map - sim_map.min())/(sim_map.max() - sim_map.min() + 1e-9)
        width, height = image.size
        sim_map = convert_similarity_to_image_size(
            sim_map, width, height, size, size)

        if viz: 
            if ax is not None:
                ax.imshow(image, cmap='gray')
                ax.imshow(sim_map, alpha=0.4, cmap='jet')
            else:
                plt.imshow(image, cmap='gray')
                plt.imshow(sim_map, alpha=0.4, cmap='jet')
                # os.makedirs(os.path.dirname(viz), exist_ok=True)
                # plt.savefig(viz, bbox_inches='tight', dpi=100)
                # plt.close()
        return sim_map

    def _get_similarity_map_from_embeddings(self,
        projected_patch_embeddings: torch.Tensor,
        projected_text_embeddings: torch.Tensor, sigma: float = 1.5
    ) -> torch.Tensor:
        """Get smoothed similarity map for a given image patch embeddings and text embeddings.

        :param projected_patch_embeddings: [n_patches_h, n_patches_w, feature_size]
        :param projected_text_embeddings: [1, feature_size]
        :return: similarity_map: similarity map of shape [n_patches_h, n_patches_w]
        """
        n_patches_h, n_patches_w, feature_size = projected_patch_embeddings.shape
        assert feature_size == projected_text_embeddings.shape[1]
        assert projected_text_embeddings.shape[0] == 1
        assert projected_text_embeddings.dim() == 2
        patch_wise_similarity = projected_patch_embeddings.view(
            -1, feature_size) @ projected_text_embeddings.t()
        patch_wise_similarity = patch_wise_similarity.reshape(
            n_patches_h, n_patches_w).cpu().detach().numpy()
        smoothed_similarity_map = torch.tensor(
            ndimage.gaussian_filter(patch_wise_similarity,
                                    sigma=(sigma, sigma), order=0)
        )
        return smoothed_similarity_map

class SoftCLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, img_labels, txt_labels):
        label_sim = torch.mm(txt_labels, img_labels.T)
        img_loss = self.soft_xent_loss(logits, F.softmax(label_sim, 1))
        txt_loss = self.soft_xent_loss(logits.t(), F.softmax(label_sim.t(), 1))
        return (img_loss+txt_loss)/2
    
    def soft_xent_loss(self, x, y):
        logprobs = F.log_softmax(x, dim=1)
        return -(y * logprobs).sum() / x.shape[0]

if __name__ == '__main__':
    from pathlib import Path
    from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
    from health_multimodal.image.data.io import load_image
    
    transform = create_chest_xray_transform_for_inference(256,256)

    
    selfeq = SelfEQ()
    
    text = ['consolidation', 'pleural effusion', # original
            'consolidation', 'pleural effusion'] # paraphrase
    input_tokens = selfeq.text_model.tokenize_input_prompts(text)
    
    img_1 = load_image(Path('./consolidation-rul-ild.jpg'))
    img_2 = load_image(Path('./effusions-bibasal.jpg'))
    
    img_t_1 = transform(img_1)
    img_t_2 = transform(img_2)
    img_tensor = torch.stack([img_t_1, img_t_2, 
                              img_t_1.detach().clone(),
                              img_t_2.detach().clone()
                              ])
    
    # img_tensor = torch.randn([4,3,256,256])
    img_labels = torch.tensor([
        [1,0,0,1],
        [0,0,0,1],
        [1,1,0,0],
        [0,0,1.,1]
    ])
    txt_labels = torch.tensor([
        [0,1.,0,1],
        [0,1,0,0],
        [1,1,0,1],
        [0,0,1,0]
    ])
    
    g, g1, g2 = selfeq(img_tensor, input_tokens, img_labels, txt_labels)
    
    # g = gradcam.mean()
    g.backward()
    # print(gradcam.shape)