import os
import torch
import trimesh
import gc
from PIL import Image
from .hy3dgen.rembg import BackgroundRemover
from .hy3dgen.texgen import Hunyuan3DPaintPipeline

class Hunyuan3DTextureMeshNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "front_image": ("IMAGE",),
            },
            "optional": {
                "left_image": ("IMAGE", {"default": None}),
                "right_image": ("IMAGE", {"default": None}),
                "back_image": ("IMAGE", {"default": None}),
                "top_image": ("IMAGE", {"default": None}),
                "bottom_image": ("IMAGE", {"default": None}),
                "remove_background": ("BOOLEAN", {"default": True}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    FUNCTION = "texture_mesh"
    CATEGORY = "mesh/texture"

    def texture_mesh(self, mesh, front_image, 
                    left_image=None, right_image=None, back_image=None, 
                    top_image=None, bottom_image=None, remove_background=True,
                    device="cuda"):
        # Free up VRAM before loading new models
        self._free_memory()
        
        # Collect all provided images
        image_inputs = [front_image]
        for img in [left_image, right_image, back_image, top_image, bottom_image]:
            if img is not None:
                image_inputs.append(img)
        
        # Convert images to PIL format
        pil_images = []
        rembg = BackgroundRemover() if remove_background else None
        
        for img in image_inputs:
            # Handle different possible image formats from ComfyUI
            if isinstance(img, Image.Image):
                pil_img = img
            elif isinstance(img, torch.Tensor):
                # Convert tensor to PIL image
                if img.ndim == 4:
                    img = img.squeeze(0)  # Remove batch dimension if present
                
                # Handle different tensor formats
                if img.shape[0] == 3 or img.shape[0] == 4:  # Channels-first format (C,H,W)
                    img_np = img.permute(1, 2, 0).cpu().numpy()
                else:  # Assume channels-last format (H,W,C)
                    img_np = img.cpu().numpy()
                
                # Scale to 0-255 range if necessary
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype('uint8')
                else:
                    img_np = img_np.astype('uint8')
                
                pil_img = Image.fromarray(img_np)
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
            
            # Process background removal if needed
            if remove_background and pil_img.mode == 'RGB':
                pil_img = rembg(pil_img)
                
            pil_images.append(pil_img)
        
        model_path = "tencent/Hunyuan3D-2"
        
        # Initialize the Hunyuan3D pipeline
        pipeline = Hunyuan3DPaintPipeline.from_pretrained(
            model_path=model_path,
        )
        
        try:
            # Apply texturing
            textured_mesh = pipeline(mesh, image=pil_images)
            return (textured_mesh,)
        finally:
            # Clean up pipeline after use to free memory
            if hasattr(pipeline, "unet"):
                del pipeline.unet
            if hasattr(pipeline, "vae"):
                del pipeline.vae
            if hasattr(pipeline, "text_encoder"):
                del pipeline.text_encoder
            del pipeline
            self._free_memory()
    
    def _free_memory(self):
        """Free GPU memory by clearing cache and collecting garbage"""
        # Empty CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Force garbage collection
        gc.collect()

# Node registration function for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Hunyuan3DTextureMesh": Hunyuan3DTextureMeshNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hunyuan3DTextureMesh": "Hunyuan3D Texture Mesh"
}