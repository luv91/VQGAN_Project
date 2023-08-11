from PIL import Image, ImageOps                   # Image Manipulation
from sklearn.decomposition import PCA
import numpy as np                     
import os
import matplotlib.pyplot as plt   
from glob import glob


def load_images_from_folder(folder_path, size=(256, 256)):
    images = []
    for img_path in glob(os.path.join(folder_path, '*.jpg')): # Assuming images are in JPG format
        img = Image.open(img_path)
        img_resized = img.resize(size) # Resize to common size
        images.append(np.array(img_resized.getdata()).reshape(*img_resized.size, -1))
    return np.stack(images)

def img_data(imgPath,disp = True):
    
    orig_img = Image.open(imgPath)
    
    img_size_kb = os.stat(imgPath).st_size/256
    
    ori_pixels = np.array(orig_img.getdata()).reshape(*orig_img.size, -1)
    
    img_dim = ori_pixels.shape 
    
    if disp:
        plt.imshow(orig_img)
        plt.show()
    
    data_dict = {}
    data_dict['img_size_kb'] = img_size_kb
    data_dict['img_dim'] = img_dim
    
    return data_dict


# imgPath = 'image_00002.jpg';

# data_dict_ori = img_data(imgPath)

# print('Original Image Data')
# print('Original Image size (kB)',data_dict_ori['img_size_kb'])
# print('Original Image Shape',data_dict_ori['img_dim'])

def pca_compose_all(images_array):
    all_pca_channels = []
    
    for img in images_array:
        # img is already a numpy array, so we don't need to read it again
        
        # Reshape 2D to 3D array, if necessary (depending on the shape of img)
        if len(img.shape) == 2:
            img = img.reshape(*img.shape, -1)
        
        # Separate channels from image and use PCA on each channel
        pca_channel = {}
        img_t = np.transpose(img) # transposing the image
        
        for i in range(img.shape[-1]):    # For each RGB channel compute the PCA
            channel = img_t[i].reshape(*img.shape[:-1])  # obtain channel
            
            pca = PCA(random_state=42)                # initialize PCA
            
            fit_pca = pca.fit_transform(channel)        # fit PCA
            
            pca_channel[i] = (pca, fit_pca)  # save PCA models for each channel
            
        all_pca_channels.append(pca_channel)

    return all_pca_channels

# folder_path = 'train_all_only_few_resized' # Change this to your folder path
# all_images = load_images_from_folder(folder_path)
# pca_channel = pca_compose_all(all_images)
# print("pca_channel", pca_channel[0].keys(), len(pca_channel))


def reconstruct_images(valid_folder_path, pca_channels, n_components, output_folder):
    all_valid_images = load_images_from_folder(valid_folder_path)
    print("all_images.shape", all_valid_images.shape)
    
    # Applying the same PCA transformations to all validation images
    for idx, valid_image in enumerate(all_valid_images):
        compressed_image = pca_transform(valid_image, pca_channels[0], n_components) # Using the same PCA model for each validation image
        image_pil = Image.fromarray(compressed_image)
        output_path = os.path.join(output_folder, f"reconstructed_image_{idx}.png")
        image_pil.save(output_path)
        print(f"Image {idx} saved to {output_path}")

def pca_transform(image, pca_channel, n_components):
    temp_res = []

    img_t = np.transpose(image) # transposing the image
    
    for i, (pca, _) in enumerate(pca_channel.values()): # iterate over the channels
        channel = img_t[i].reshape(*image.shape[:-1])  # obtain channel
        fit_pca = pca.transform(channel)  # transform using the PCA model
        
        # Selecting image pixels across first n components
        pca_pixel = fit_pca[:, :n_components]
        
        # First n-components
        pca_comp = pca.components_[:n_components, :]
        
        # Projecting the selected pixels along the desired n-components (De-standardization)
        compressed_pixel = np.dot(pca_pixel, pca_comp) + pca.mean_
        
        # Stacking channels corresponding to Red, Green, and Blue
        temp_res.append(compressed_pixel)
            
    # transforming (channel, width, height) to (height, width, channel)
    compressed_image = np.transpose(temp_res, (1, 2, 0))

    # Forming the compressed image
    compressed_image = np.array(compressed_image, dtype=np.uint8)
    
    return compressed_image

folder_path = 'train_all_2700_resized' # Training images folder path
valid_folder_path = 'valid_all_2700_resized' # Validation images folder path
output_folder = 'valid_all_2700_resized_reconstructed_images' # Output folder for reconstructed images
os.makedirs(output_folder, exist_ok=True)
n_components = 50

# Load training images and perform PCA decomposition
all_images = load_images_from_folder(folder_path)
pca_channel = pca_compose_all(all_images)

# Reconstruct validation images using the PCA models
# reconstruct_images(valid_folder_path, pca_channel, n_components, output_folder)

print("Reconstruction completed successfully!")

def explained_var_n(pca_channels, n_components):
    var_exp_channel = []
    var_exp = 0
    
    for pca_channel in pca_channels:
        for channel_key, (pca, _) in pca_channel.items():
            
            # Ensure n_components does not exceed the length of explained_variance_ratio_
            n = min(n_components, len(pca.explained_variance_ratio_) - 1)
            
            var_exp_channel_val = np.cumsum(pca.explained_variance_ratio_)[n]
            var_exp_channel.append(var_exp_channel_val)
            
            var_exp += var_exp_channel_val
        
    var_exp = var_exp / len(var_exp_channel)
    
    return var_exp
# Let's choose a random number of component to retain
# As we know, our image is of the shape 1024*1024 so maximum number of components can be 0 to 1023.
# Let's see:

var_exp = explained_var_n(pca_channel, 50)
print("Explained variance in percentage by PCA : ", var_exp*100,"%")

def variance_added_pc(pca_channels):
    var_exp_channels = []

    # Iterate through the PCA models for each image
    for pca_channel in pca_channels:
        var_exp_channel = []
        # Iterate through the PCA models for each channel in the image
        for _, (pca, _) in pca_channel.items():
            var_exp_channel.append(pca.explained_variance_ratio_)
        
        # Average the variance explained across the channels for this image
        var_exp_channels.append(np.mean(var_exp_channel, axis=0))

    # Average the variance explained across all the images
    var_exp = np.mean(var_exp_channels, axis=0)

    x = var_exp[:50]
    y = range(1, 51)

    plt.yticks(np.arange(0, max(x) + 0.05, 0.05))
    plt.xticks(np.arange(min(y), max(y) + 1, 1))
    plt.title("Individual Variance for each Principal Component")
    plt.ylabel('Variance')
    plt.xlabel('Principal Component')
    plt.bar(y, x, color='black')
    plt.show()

variance_added_pc(pca_channel)

# Function to plot the explained variance/information w.r.t the number of components

def plot_variance_pc(pca_channels):
    exp_var = {}
    
    for i in range(len(pca_channels[0][0][0].components_)):
        var_exp = explained_var_n(pca_channels, i)
        exp_var[i+1] = var_exp
    
    lists = sorted(exp_var.items())
    x, y = zip(*lists)
    
    pt90 = next(xx[0] for xx in enumerate(y) if xx[1] > 0.9)
    pt95 = next(xx[0] for xx in enumerate(y) if xx[1] > 0.95)
    
    plt.plot(x, y)
    plt.vlines(x=x[pt90], ymin=0, ymax=y[pt90], colors='green',  ls=':', lw=2, label=str('90% Variance Explained : n = '+str(x[pt90])))
    plt.vlines(x=x[pt95], ymin=0, ymax=y[pt95], colors='red', ls=':', lw=2, label=str('95% Variance Explained : n = '+str(x[pt95])))
    
    plt.xticks(np.arange(min(x)-1, max(x)-1,100))
    plt.yticks(np.arange(0, max(y),0.1))
    
    plt.legend(loc="lower right")
    plt.title("Variance vs Principal Components")
    plt.xlabel("Principal Components")
    plt.ylabel("Variance Explained")
    plt.grid(True)
    plt.show()
plot_variance_pc(pca_channel)



# Function to tell the percentage of explained variance by n number of components

# def explained_var_n(pca_channel, n_components):
#     var_exp_channel = []
#     var_exp = 0
    
#     for channel in pca_channel:
#         pca, _ = pca_channel[channel]
        
#         # Ensure n_components does not exceed the length of explained_variance_ratio_
#         n = min(n_components, len(pca.explained_variance_ratio_) - 1)
        
#         var_exp_channel.append(np.cumsum(pca.explained_variance_ratio_))
#         var_exp += var_exp_channel[channel][n]
        
#     var_exp = var_exp / len(pca_channel)
    
#     return var_exp
# # Let's choose a random number of component to retain
# # As we know, our image is of the shape 1024*1024 so maximum number of components can be 0 to 1023.
# # Let's see:

# var_exp = explained_var_n(pca_channel, 1023)

# print("Explained variance in percentage by PCA : ", var_exp*100,"%")

# # Function to plot the individual variance of every principal component
# def variance_added_pc(pca_channel):
    
#     var_exp_channel = [];var_exp=0;
    
#     for channel in pca_channel:
#         pca,_ = pca_channel[channel]
#         var_exp_channel.append(pca.explained_variance_ratio_)
        
#     var_exp = (var_exp_channel[0]+var_exp_channel[1]+var_exp_channel[2])/3
    
#     x = list(var_exp);y = list(range(1,1+len(x)));y = list(range(1,21))
    
#     plt.yticks(np.arange(0, max(x)+0.05,0.05))
#     plt.xticks(np.arange(min(y), max(y)+1,1))
#     plt.title("Individual Variance for each Principal Component")
#     plt.ylabel('Variance')
#     plt.xlabel('Principal Component');
#     plt.bar(y,x[:20],color = 'black')
#     #plt.grid(True)
#     plt.show()
    
# variance_added_pc(pca_channel)

# # Function to plot the explained variance/information w.r.t the number of components

# def plot_variance_pc(pca_channel):
    
#     pca,fit_pca = pca_channel[0]
    
#     exp_var = {}
    
#     for i in range(len(pca.components_)):
#         var_exp = explained_var_n(pca_channel,i)
#         exp_var[i+1] = var_exp
    
#     lists = sorted(exp_var.items()) # sorted by key, return a list of tuples
    
#     x, y = zip(*lists) # unpack a list of pairs into two tuples
    
#     pt90 = next(xx[0] for xx in enumerate(y) if xx[1] > 0.9)
#     pt95 = next(xx[0] for xx in enumerate(y) if xx[1] > 0.95)
    
#     plt.plot(x, y)
#     plt.vlines(x=x[pt90], ymin=0, ymax=y[pt90], colors='green',  ls=':', lw=2, label=str('90% Variance Explained : n = '+str(x[pt90])))
#     plt.vlines(x=x[pt95], ymin=0, ymax=y[pt95], colors='red', ls=':', lw=2, label=str('95% Variance Explained : n = '+str(x[pt95])))
    
#     plt.xticks(np.arange(min(x)-1, max(x)-1,100))
#     plt.yticks(np.arange(0, max(y),0.1))
    
#     plt.legend(loc="lower right")
#     plt.title("Variance vs Principal Components")
#     plt.xlabel("Principal Components")
#     plt.ylabel("Variance Explained")
#     plt.grid(True)
#     plt.show()
    
    
# # So even if we select 50 components, the information retention will be huge.
# n_components = 50

# var_exp = explained_var_n(pca_channel, n_components)

# print("Explained variance in percentage by PCA : ", var_exp*100,"%")


# def pca_transform(pca_channel, n_components):
    
#     temp_res = []
    
#     # Looping over all the channels we created from pca_compose function
    
#     for channel in range(len(pca_channel)):
        
#         pca, fit_pca = pca_channel[channel]
        
#         # Selecting image pixels across first n components
#         pca_pixel = fit_pca[:, :n_components]
        
#         # First n-components
#         pca_comp = pca.components_[:n_components, :]
        
#         # Projecting the selected pixels along the desired n-components (De-standardization)
#         compressed_pixel = np.dot(pca_pixel, pca_comp) + pca.mean_
        
#         # Stacking channels corresponding to Red Green and Blue
#         temp_res.append(compressed_pixel)
            
#     # transforming (channel, width, height) to (height, width, channel)
#     compressed_image = np.transpose(temp_res)
    
#     # Forming the compressed image
#     compressed_image = np.array(compressed_image,dtype=np.uint8)
    
#     return compressed_image

# compressed_image = pca_transform(pca_channel,n_components=n_components)

# # Display

# plt.imshow(compressed_image)
# plt.show()

# Image.fromarray(compressed_image).save("compressed_img.jpg")

# This will save the compressed image