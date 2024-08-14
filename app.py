from diffusers import StableDiffusionPipeline
import os
import torch  # Certifique-se de importar torch no início

# Ajustar configuração para gerenciar melhor a memória
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

model_id = "runwayml/stable-diffusion-v1-5"

try:
    # Tentar carregar o pipeline com menos memória
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to("cuda")  # Tente usar a GPU se houver memória suficiente

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]
    image.save("astronaut_rides_horse.png")

except torch.cuda.OutOfMemoryError:
    print("CUDA out of memory. Switching to CPU.")
    # Tentar usar a CPU se não houver memória GPU suficiente
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to("cpu")

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]
    image.save("astronaut_rides_horse.png")

finally:
    torch.cuda.empty_cache()  # Limpar cache da GPU
