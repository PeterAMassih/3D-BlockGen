
# TODO write last, + no citations on the abstract
# Do not forget to cite
1. Abstract

Motivation: Need for automated 3D model generation
Background on diffusion models
Project goals: Text-to-3D generation using diffusion models
Challenges in 3D voxel generation
time-consuming nature of manual LEGO®
design by automating the generation process through a
multi-stage diffusion pipeline (Two stage)

2. Introduction talk rapid about the work that will come. 
We make this dataset + code public leak the github + huggingface doi
show some inference plot, from text to shape to shape with color
Say what what will come in a fast manner


3. Related Work
Diffusion models in 3D generation
Prior work in text-to-3D
First paper related: XCube: Large-Scale 3D Generative Modeling using Sparse Voxel Hierarchies
Second paper: DiT-3D: Exploring Plain Diffusion Transformers for 3D Shape Generation
Third a discussion on papers: diffusion generation tasks of IV 3D Diffusion Generation Tasks
Fourth: DiffRF: Rendering-Guided 3D Radiance Field Diffusion
Fifth: Scaling Diffusion Models to Real-World 3D LiDAR Scene Completion 


4. Background
Technical background on diffusion models some equations to reintroduce DDPIM and DDPMS quickly
Show some technicalities and calculations in md math

5. -- Methodology -- Where each section is a subsection

5.1 Data Processing Pipeline

Objaverse data to introduce what it is, and say how did we use it,
collection and filtering,

Voxelization process using trimesh talk about subdivide and the flood fill algorithms

Color handling:

Material colors
UV mapping for color
Default color handling -> grey when no color

Data augmentation with rotations

Show visualization of before and after voxelization

Dataset statistics and analysis, also distribution per class/category, heatmaps of where the most occupancy is to draw a link between model output and this after but here just show these.

How did we get the prompt from the 3d models ? category + tag + what we need


5.2 Model Architecture


UNet3D structure from HuggingFace explain what it is # 2d conv along x,y 1d along z axis -> also limitation
not enough 3d capability along z limitation but if we do conv3d we use too much memory talk about ur model architecture (that i tackled but too heavy on memory). Workaround to have a model that fits in GPU memory.
Modifications for 3D generation num channels as the time dimension, frames to get the z dimensions
So explain the model architecture of this Unet.

Multi-stage approach: (also talk about input output channel on each stage)

Combined RGBA model (2 versions simple MSE + special Combined Loss)
Two-stage pipeline (MSE on occupancy + MSE special loss on color)


Loss functions and mathematical formulation
EMA implementation

Loss functions:

Color stage MSE loss
staged loss, MSE On occupancy in first stage then MSE on the color on second stage.

5.3 Inference Post-processing

Classifier-free guidance implementation cite google
Mean rotation handling, another inference strategy

5.4 LegoLization Algorithm

Brick size choosing
Greedy placement algorithm
Not optimal, do not take account to physical constraint
Color averaging
Visualization with some of the constraint we have put in our algo (maybe visualize a sphere or something simple)

--- End Methodology ---

6. Results & Analysis

Configuration details - hardware, learning rates, batch, Optimizer Cosine, Scheduler, Adamw
-- Justify quickly what we have (for example batch size until GPU vram full lr common for models with attention... maybe use a paper ?)

How did we train text ? Prompt 10% no prompt 45% detailed 45% simple, explain these , cite the OpenAI DALL-E paper here + show word cloud of prompts

Quantitative metrics:
Show losses, train + test talk about the fit

<!-- Visualization of Augmentations:
Show the 4 augmentations

Visualization of the data pipeline:
Maybe refer to prior visualization -->

How to quantify and differentiate the different models, IoU jaccard + similarity for colors (as we said before)
Show which version of the model is the best in here. (combined MSE vs combined with combined loss vs two_stage at 60k training step) then choose winner (should be two_stage),

Then on the two_stage model start with the ablation studies:

Ablation studies - finetune vs not finetuned of the winning model (two_stage), guidance scaling test some guidance scales, rotation or not, mean_init or not, DDPM or DDIM see if DDIM make me loose expressivity and worst results, EMA vs no EMA

Show also : How the diffusion process looks like at each step - gif or mutliple steps showcasing in the report still to code

end on Failure cases - Hard prompt -> bad output "Grey dog jumping on his owner"
2 objects, complex scene



7. Ideas & Future Work

Summary/Conclusion 
Limitations talk about the hard prompts, say limited or if have fixes
Potential improvements - Image cond + Plus grand dataset, finetuning avec meilleur prompt de couleur
Future research directions
futur work: annotation model positive score if the this model says if it is a banana or not. worth mentioning in futur work, if we have more robust model can do this automated metric to quantify model output.
for now we assume everything in a 3d grid strong assumption 99% of true lego models , don't only use rectangle bricks and not everything in a grid that is what we need to do to explore the subject

Remove the difference of physical constraint but instead train from scratch, link a little bit better over my work.

Discuss here:

We will discuss future improvements including: using Zero123-XL Stable diffusion + NERF trained to first generate high-quality 3D meshes that we can then discretize into our 32³ voxel format for better geometric understanding, integrating the LDRAW dataset to teach our model valid LEGO® construction patterns through fine-tuning (since LDRAW provides exact specifications for brick arrangements like defining a 2x2 brick with precise stud placements), and exploring memory optimization techniques given our current training limitation of 4 samples per batch when working with RGBA channels in our 32³ voxel grid.
LDRAW is more text type of input rather than visualize, one other approach treat this LDRAW as a programming language and finetune a LLM, but still need some notion to what it looks like.
Use CLIP to give image prompt instead of text (or together).
+ annotation model in order to provide some kind of metric instead of visually deciding.
Also we can say that can do a FID score, for 3d though.


<!-- -- DETAILS --

1.Related Work & Background

This section will introduce the theoretical foundations of diffusion models, contextualizing our work within existing text-to-3D generation approaches. We'll discuss the current limitations of automated LEGO® design tools and the gap our work fills. We'll also explain why we chose Objaverse as our dataset and compare it with alternatives?.

2.Data Processing Pipeline

Here we'll do a deep dive into our data preparation approach, starting with data retrieval from Objaverse using a custom downloader, explaining our voxelization process including how we handle mesh holes using flood-fill, and detailing our color processing pipeline that handles UV maps, material colors, and default colors. We'll share specific statistics about our dataset (135k objects with 4 augmentations each) and explain our prompt engineering strategy to obtain better results. + Subdivide also speak about this.

3.Model Architecture

This section will detail our UNet3D model from HuggingFace Diffusers, explaining modifications we made to handle both occupancy and RGBA data. We'll present mathematical formulations of our loss functions (BCE for shape why we did not use this and used MSE, MSE for color) and explain our implementation of EMA for stability. The section will thoroughly describe our three training approaches: shape-only, combined RGBA, and two-stage pipeline.

4.Training Methodology

We'll describe our training process in detail, including our use of diffusion schedulers, how we handle the CLIP embeddings for text conditioning, and our implementation of different loss functions. We'll provide specifics about hyperparameters, batch sizes, and training duration for each stage.

5.Inference Pipeline

This section will explain our inference process, including classifier-free guidance implementation, handling of mean rotation, and our visualization approaches using both matplotlib and PyTorch3D. We'll discuss how we handle prompt engineering during inference and our method for generating "high-quality" :) samples.

6.LegoLization Algorithm

Here we'll present our algorithm for converting voxel outputs into constructible LEGO® designs. We'll explain our brick size library (2x4, 2x3, etc.), our greedy placement algorithm, support constraints, and color averaging approach. We'll include visualizations of the conversion process and discuss how we ensure physical buildability.

7.Results & Analysis

This section will present both quantitative and qualitative results. We'll show example generations, discuss success cases and failure modes, and provide analysis of different model configurations. We'll include ablation studies on different components of our data pipeline mainly AKA finetune vs not finetuned, data parameter.

8.Discussion & Future Work

Finally, we'll discuss the implications of our work, current limitations (like DMD memory constraints and processing speed), and potential future improvements. We'll also suggest directions for future research in this area.
We will discuss future improvements including: using Zero123-XL Stable diffusion + NERF trained to first generate high-quality 3D meshes that we can then discretize into our 32³ voxel format for better geometric understanding, integrating the LDRAW dataset to teach our model valid LEGO® construction patterns through fine-tuning (since LDRAW provides exact specifications for brick arrangements like defining a 2x2 brick with precise stud placements), and exploring memory optimization techniques given our current training limitation of 4 samples per batch when working with RGBA channels in our 32³ voxel grid.
LDRAW is more text type of input rather than visualize, one other approach treat this LDRAW as a programming language and finetune a LLM, but still need some notion to what it looks like.
Use CLIP to give image prompt instead of text (or together). -->
