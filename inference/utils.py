def get_keypoints(heatmap):
    



def draw_pose(draw, heatmap, box):
    """
    Arguments:
        draw: an instance of ImageDraw.Draw.
        keypoints: a numpy float int with shape [17, 3]. 
    """
    x0, y0 = origin
    for j in range(17):
        mask = binary_masks[:, :, j]
        if mask.max() > 0:
            y, x = np.unravel_index(mask.argmax(), mask.shape)
def draw_everything(image, outputs):

    image_copy = image.copy()
    image_copy.putalpha(255)
    draw = ImageDraw.Draw(image_copy, 'RGBA')
    width, height = image_copy.size
    scaler = np.array([height, width, height, width])
    
    n = outputs['num_boxes']
    boxes = scaler * outputs['boxes']
    for box in boxes:
        ymin, xmin, ymax, xmax = box
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='red')
    
    mask = outputs['keypoint_heatmaps'][:, :, 0]#outputs['segmentation_masks']
    m, M = mask.min(), mask.max()
    mask = (mask - m)/(M - m)
    mask = np.expand_dims(mask, 2)
    color = np.array([255, 255, 255])
    mask = Image.fromarray((mask*color).astype('uint8'))
    mask.putalpha(mask.convert('L'))
    mask = mask.resize((width, height))
    image_copy.alpha_composite(mask)
    return image_copy