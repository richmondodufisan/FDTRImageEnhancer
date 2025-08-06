import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image

def cluster_grain_regions(image_path, n_clusters=3, save_path="clustered_output_image.png"):
    
    """
    Load an image, apply KMeans clustering to color regions, 
    and output segmented image and cluster info.
    """
    
    # Load image and convert to RGB, becomes object of (height, width, 3)
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Reshape to (num_pixels, 3) for clustering
    pixels = image_np.reshape(-1, 3)

    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(pixels)
    cluster_centers = kmeans.cluster_centers_.astype(np.uint8)

    # Reconstruct segmented image
    segmented_img = cluster_centers[labels].reshape(image_np.shape)
    region_map = labels.reshape(image_np.shape[0], image_np.shape[1])

    # Print cluster info
    print(f"\nNumber of clusters detected: {len(np.unique(labels))}")
    print("Cluster center RGB values:")
    for i, center in enumerate(cluster_centers):
        print(f"Cluster {i}: {center}")

    # Save and show images
    Image.fromarray(segmented_img).save(save_path)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(image_np)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(segmented_img)
    axs[1].set_title("Segmented (K-Means)")
    axs[1].axis("off")

    plt.tight_layout()
    plt.savefig("clustered_v_original.png", dpi = 600)
    plt.show()
    
    region_map = np.flipud(region_map)  # consistent axis
    np.save("region_map.npy", region_map)

    return region_map, cluster_centers

# Run
if __name__ == "__main__":
    image_path = "./grain_structure.png"  # Change to your filename
    cluster_grain_regions(image_path, n_clusters=2)  # Use 2 if you just want GB vs Bulk


