# Estimating-LC-orientation-with-AI
2D apporach for the prediction\
Can only predict one layer of the director field of LC\
Also no out-of-plane involved\
CNN_mk4.py is the latest version\
Chnage path folder in the CNN code\
The target folder should have three subfolder: train, val, test (must be in lowercase letter)\
If train/val/test split is 70%/15%/15%, and 1000 sample images, then train folder should have 750 images and so on.\

Step\
1. Read instruction.txt and generate sample images (or generate sample with other methods)\
2. Update the folder path of the CNN code\
3. Run with 5 epochs to see if it has no errors (optional but recommand)\
4. Throw the samples and CNN code into HPC\
5. Run CNN on HPC and fine-tune the parameter (i.e. epochs, learning rate "lr" of the optimizer)
