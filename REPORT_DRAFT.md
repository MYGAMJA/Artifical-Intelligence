# Assignment 3 Report Draft

## Experiment Setup

This assignment implements Textual Inversion for Stable Diffusion personalization. I used the `cat2` reference images as the target concept. The placeholder token was `<cat2>`, and the initializer token was `cat`. The goal was to optimize only the embedding of the new placeholder token while keeping the pretrained Stable Diffusion components frozen.

| Item | Setting |
| --- | --- |
| Concept | `cat2` |
| Placeholder token | `<cat2>` |
| Initializer token | `cat` |
| Learnable property | object |
| Training steps | 1000 |
| Save interval | 100 steps |
| Batch size | 4 |
| Mixed precision | fp16 |

## Problem 1. Learning the New Token

### 1. Tokenizer and Initializer Token

![Problem 1 tokenizer TODO](report_images/problem1_tokenizer.png)

This part adds the new placeholder token `<cat2>` to the CLIP tokenizer. Since Textual Inversion represents a new concept through a newly introduced token, the placeholder token must not already exist in the vocabulary. If the token already exists, the code raises an error to prevent accidentally overwriting an existing word representation.

The initializer token `cat` is also encoded and checked to make sure that it is represented as a single token. This is important because the new placeholder embedding is initialized from the embedding of `cat`. Since the target concept is a cat, this gives `<cat2>` a meaningful semantic starting point instead of starting from a random embedding.

### 2. Placeholder Embedding Initialization

![Problem 1 embedding initialization](report_images/problem1_embedding_init.png)

After the tokenizer is expanded, the text encoder embedding layer is resized to match the new tokenizer length. The embedding vector of `<cat2>` is then initialized by copying the embedding vector of `cat`.

This initialization helps training because the new token already begins near the general visual concept of a cat. The optimization then only needs to move the embedding from the generic class concept toward the specific reference subject.

### 3. Freezing the Model

![Problem 1 freeze TODO](report_images/problem1_freeze.png)

The VAE and U-Net are frozen because Textual Inversion does not fine-tune the full Stable Diffusion model. In the text encoder, the transformer encoder, final layer normalization, and position embedding are also frozen.

This leaves only the input token embedding layer trainable. Freezing the main model is the key idea of Textual Inversion: the pretrained model keeps its original image generation ability, while the new concept is stored in a small learned token embedding.

### 4. Optimizer Parameter Selection

![Problem 1 optimizer TODO](report_images/problem1_optimizer.png)

The optimizer receives the text encoder input embedding parameters. This is the only part that needs to be optimized for learning the placeholder token.

Although the embedding layer contains embeddings for all tokens, the code later masks out the gradients of every token except `<cat2>`. Therefore, the optimizer is technically connected to the embedding matrix, but only the placeholder token embedding is actually updated.

### 5. Forward Diffusion Process

![Problem 1 forward diffusion TODO](report_images/problem1_forward_diffusion.png)

The reference images are first encoded into latent representations using the VAE encoder. The latents are then multiplied by the Stable Diffusion scaling factor so that their scale matches the diffusion model's expected latent space.

Next, random Gaussian noise and random timesteps are sampled. The scheduler adds the sampled noise to the latents according to the selected timesteps. This creates noisy latent inputs, which simulate the forward diffusion process used during diffusion model training.

### 6. Reverse Diffusion Process

![Problem 1 reverse diffusion TODO](report_images/problem1_reverse_diffusion.png)

The text prompt containing `<cat2>` is passed through the text encoder to obtain text conditioning. The U-Net then receives the noisy latents, the sampled timesteps, and the text encoder hidden states.

The U-Net predicts the noise contained in the noisy latents. Since the text conditioning includes the learnable placeholder token, the prediction error provides a training signal for the `<cat2>` embedding.

### 7. MSE Loss and Gradient Masking

![Problem 1 loss and gradient masking](report_images/problem1_loss_gradient_mask.png)

The loss is computed as mean squared error between the predicted noise and the target noise. In this setting, the target is the original Gaussian noise that was added to the latent.

After backpropagation, gradient masking is applied to the embedding matrix. All token gradients except the placeholder token gradient are set to zero. This step is essential because the method should not change the embeddings of existing vocabulary tokens. As a result, only the `<cat2>` embedding is updated.

### Problem 1 Summary

In Problem 1, I added a new placeholder token, initialized it from the semantically related token `cat`, froze the pretrained diffusion model components, and optimized only the new token embedding. The training loop follows the standard diffusion noise prediction objective, but the actual parameter update is restricted to the placeholder embedding.

## Problem 2. Use and Evaluate

### 1. Image Generation with the Learned Token

![Problem 2 inference code](report_images/problem2_inference_code.png)

After training, I loaded the learned embedding from the selected checkpoint and generated images using the prompt:

`a photo of a <cat2> in the desert`

This experiment checks whether the learned token can preserve the personalized concept while placing it in a new scene. The reference images do not contain this desert background, so successful generation means that the learned token can be recombined with the pretrained model's existing knowledge.

![Problem 2 desert results](report_images/problem2_desert_results.png)

The generated samples generally follow the desert prompt and produce a cat-like subject. The model captures the broad object category and can place it in a new environment. However, some fine details of the reference cat may vary across samples, which shows that a single learned token embedding does not perfectly preserve every identity detail.

### 2. Denoising Process Visualization

![Problem 2 denoising callback](report_images/problem2_denoising_callback.png)

The main TODO in Problem 2 implements a callback that saves intermediate images during inference. The pipeline runs for 30 inference steps, and 15 evenly spaced steps are selected for visualization.

At each selected step, the current latent is decoded into an image. Before decoding, the latent is divided by the VAE scaling factor because Stable Diffusion stores images in a scaled latent space. The decoded tensor is then normalized to the image range `[0, 1]` and converted into a PIL image.

![Problem 2 denoising results](report_images/problem2_denoising_results.png)

The denoising sequence shows how the image evolves from noise to a final sample. In the early steps, the image mostly contains noise and rough global composition. In the middle steps, the main object shape becomes clearer. In the final steps, details such as edges, texture, and background structure are refined.

This supports the observation that diffusion generation first forms low-frequency structure such as layout and color, and later refines high-frequency details.

### 3. CLIP-I and CLIP-T Evaluation

![Problem 2 CLIP score code](report_images/problem2_clip_code.png)

For quantitative evaluation, I used CLIP-I and CLIP-T. CLIP-I measures visual similarity between the generated image and a reference image. CLIP-T measures semantic alignment between the generated image and the text prompt.

One important detail is that the generated image and prompt must match when computing CLIP-T. For example, an image generated with the desert prompt should also be evaluated with the desert prompt, not the cobblestone street prompt.

| Prompt | Embedding step | CLIP-I | CLIP-T | Observation |
| --- | ---: | ---: | ---: | --- |
| `a photo of a <cat2> in the desert` | 1000 |  |  |  |
| `a photo of a <cat2> on a cobblestone street` | 1000 |  |  |  |
| `an oil painting of a <cat2>` | 1000 |  |  |  |

### Problem 2 Summary

The learned `<cat2>` token can guide Stable Diffusion to generate the personalized concept in new prompts. The denoising visualization shows that the model gradually builds the image from global structure to local details. The qualitative results and CLIP metrics together show a tradeoff: the learned token preserves the general identity of the concept, but fine-grained details can weaken when the prompt changes the scene or style too much.

## Problem 3. Additional Experiments

### 1. Additional Experiment Code

![Problem 3 experiment code](report_images/problem3_experiment_code.png)

For Problem 3, I designed additional experiments to check when Textual Inversion succeeds and when it fails. I used the saved training runs in the workspace:

| Folder | Learning rate label |
| --- | --- |
| `sd-concept-output_1` | `lr0.002` |
| `sd-concept-output_2` | `lr0.0004` |
| `sd-concept-output_3` | `lr0.004` |

The experiments compare three factors: training step, learning rate, and prompt difficulty. The code saves generated images and computes CLIP-I and CLIP-T for each condition.

### 2. Step Comparison

![Problem 3 step comparison results](report_images/problem3_step_results.png)

I compared checkpoints from step 200, 500, and 1000 using the same prompt. This experiment shows how the learned embedding changes during training.

Early checkpoints are expected to behave closer to the generic initializer token `cat`, so they may produce a normal cat but not the specific target identity. Later checkpoints should better capture the reference concept. However, too much training can also reduce flexibility if the embedding overfits to the reference images.

| Step | Prompt | CLIP-I | CLIP-T | Observation |
| ---: | --- | ---: | ---: | --- |
| 200 | `a photo of a <cat2> in the desert` |  |  |  |
| 500 | `a photo of a <cat2> in the desert` |  |  |  |
| 1000 | `a photo of a <cat2> in the desert` |  |  |  |

### 3. Learning Rate Comparison

![Problem 3 learning rate results](report_images/problem3_lr_results.png)

I compared the final checkpoint from three learning-rate runs. A smaller learning rate should train more slowly but may be more stable. A larger learning rate can make the embedding change quickly, but it may also distort the concept or reduce prompt controllability.

This experiment helps show that Textual Inversion is sensitive to training hyperparameters even though only one token embedding is being optimized.

| Run | Prompt | CLIP-I | CLIP-T | Observation |
| --- | --- | ---: | ---: | --- |
| `lr0.0004` | `a photo of a <cat2> in the desert` |  |  |  |
| `lr0.002` | `a photo of a <cat2> in the desert` |  |  |  |
| `lr0.004` | `a photo of a <cat2> in the desert` |  |  |  |

### 4. Prompt Robustness

![Problem 3 prompt comparison results](report_images/problem3_prompt_results.png)

I tested the same learned embedding with prompts of different difficulty:

| Prompt type | Prompt |
| --- | --- |
| Simple scene | `a photo of a <cat2> in the desert` |
| New background | `a photo of a <cat2> on a cobblestone street` |
| Style change | `an oil painting of a <cat2>` |
| Hard composition | `a <cat2> and a dog` |

The model is expected to work better when the prompt contains a single object and a simple background. It may become less stable when the prompt asks for strong style transfer or interaction with another object. This suggests that one embedding vector has limited capacity for representing both detailed identity and flexible composition.

### 5. Improvement Proposal

Based on the experiments, I propose the following improvements:

- Use more diverse reference images with different poses and backgrounds.
- Apply moderate data augmentation such as random crop or horizontal flip.
- Tune the learning rate carefully to avoid under-training or overfitting.
- Use multiple learned tokens if one embedding is not enough to represent detailed identity.
- Add regularization so that the learned embedding does not drift too far from the initializer token.

### Problem 3 Summary

The additional experiments show that Textual Inversion is efficient but sensitive to training settings. It performs well for simple object personalization, but it struggles when the prompt requires detailed identity preservation, strong style change, or complex object interaction. The main limitation is that only the token embedding is optimized, so the representation capacity is much smaller than methods that fine-tune more model parameters.
