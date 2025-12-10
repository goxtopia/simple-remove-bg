import cv2
import numpy as np

def process_image(image_bytes: bytes, threshold: int = 20) -> tuple[bytes, np.ndarray, np.ndarray]:
    """
    Removes the background from an image using Flood Fill.

    Args:
        image_bytes: The input image as bytes.
        threshold: The threshold for flood fill tolerance.

    Returns:
        A tuple containing:
        - The processed image as PNG bytes.
        - The original image (numpy array).
        - The processed image (numpy array).
    """
    # Decode the image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Could not decode image")

    # Add a 1-pixel border to the image to ensure corner connectivity
    # This helps if the object touches the edge but not the corners
    h, w = image.shape[:2]
    padded_image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    # Create a mask for floodFill (must be 2 pixels larger than image)
    h_pad, w_pad = padded_image.shape[:2]
    mask = np.zeros((h_pad + 2, w_pad + 2), np.uint8)
    
    # Floodfill from (0,0) - top left corner of the padded image
    # Note: padded_image is 1px larger than original, so (0,0) is the added border
    seed_point = (0, 0)
    
    # Flags:
    # 4 connectivity
    # FLOODFILL_MASK_ONLY: We only want to update the mask, not the image yet
    # (255 << 8): Fill value for the mask (255)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    
    # Determine the seed color (the color of the border/background)
    # Let's trust the corner pixel of the *original* image.
    
    # Better approach:
    # 1. Pad the image with the color of the top-left pixel of the original image.
    corner_color = image[0, 0].tolist() 
    padded_image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=corner_color)
    
    # 2. Flood fill
    cv2.floodFill(padded_image, mask, seed_point, (0, 0, 0), (threshold,)*3, (threshold,)*3, flags)
    
    # The mask now has 255 where the background is.
    # The mask size is (h+2+2, w+2+2) relative to padded_image?
    # No, mask is (h_pad + 2, w_pad + 2).
    # The interesting part of the mask corresponds to padded_image is mask[1:-1, 1:-1]
    
    # Crop the mask to match the padded image size
    # mask includes the extra 1px border required by floodFill
    flood_mask = mask[1:-1, 1:-1]
    
    # Remove the 1px padding we added to the image and mask
    final_mask = flood_mask[1:-1, 1:-1]
    
    # Invert mask: Background was 255, so Foreground is 0. 
    # We want Foreground = 255 (Opaque), Background = 0 (Transparent)
    # So valid area is where mask is 0.
    alpha_channel = np.where(final_mask == 255, 0, 255).astype(np.uint8)
    
    # Create RGBA image
    b, g, r = cv2.split(image)
    rgba = cv2.merge([b, g, r, alpha_channel])
    
    # Crop to the bounding box of the non-transparent area
    # Find contours or just use numpy
    y_indices, x_indices = np.nonzero(alpha_channel)
    
    if len(y_indices) > 0:
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        
        cropped_rgba = rgba[y_min:y_max+1, x_min:x_max+1]
    else:
        # If everything is removed, return empty or full transparent
        cropped_rgba = rgba
        
    # Encode back to PNG
    success, encoded_image = cv2.imencode('.png', cropped_rgba)
    if not success:
        raise ValueError("Could not encode image")
        
    return encoded_image.tobytes(), image, cropped_rgba
