# Function definitions 
import cv2 as cv
import numpy as np

def getLargestContour(img_BW):
    """ Return largest contour in foreground as an nd.array """
    contours, hier = cv.findContours(img_BW.copy(), cv.RETR_EXTERNAL,
                                     cv.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv.contourArea)
    
    return np.squeeze(contour)

def getConvexityDefects(contour):
    """ Return convexity defects in a contour as an nd.array """
    hull = cv.convexHull(contour, returnPoints=False)
    defects = cv.convexityDefects(contour, hull)
    if defects is not None:
        defects = defects.squeeze()

    return defects

def getAdvancedContourFeatures(contour):
    """ Return advanced contour features based on contour analysis. """
    # Simple contour features
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    x, y, w, h = cv.boundingRect(contour)
    aspect_ratio = float(w) / h
    rect_area = w * h
    extent = float(area) / rect_area

    # Convex hull features
    hull = cv.convexHull(contour)
    hull_area = cv.contourArea(hull)
    hull_perimeter = cv.arcLength(hull, True)
    solidity = float(area) / hull_area if hull_area > 0 else 0

    # Circularity
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    # Eccentricity and fitting ellipse parameters
    try:
        ellipse = cv.fitEllipse(contour)
        (center, axes, orientation) = ellipse
        major_axis_length = max(axes)
        minor_axis_length = min(axes)
        eccentricity = np.sqrt(1 - (minor_axis_length / major_axis_length) ** 2) if major_axis_length > 0 else 0
    except:
        major_axis_length = minor_axis_length = orientation = eccentricity = 0

    # Convexity defects
    hull_indices = cv.convexHull(contour, returnPoints=False)
    defects = cv.convexityDefects(contour, hull_indices)
    num_convexity_defects = defects.shape[0] if defects is not None else 0
    max_defect_depth = max(defects[:, 0, 3]) / 256.0 if defects is not None else 0  # Normalize depth
    
    moments = cv.moments(contour)
    huMoments = cv.HuMoments(cv.moments(contour)).flatten()
    scaled_huMoments = -1.0 * np.sign(huMoments) * np.log10(abs(huMoments))
    
    # Centroid calculation
    centroid_x = int(moments['m10'] / moments['m00']) if moments['m00'] > 0 else 0
    centroid_y = int(moments['m01'] / moments['m00']) if moments['m00'] > 0 else 0
    centroid = (centroid_x, centroid_y)
    
    # Create a feature vector
    features = np.array([
        area,
        perimeter,
        aspect_ratio,
        extent,
        hull_area,
        hull_perimeter,
        solidity,
        circularity,
        eccentricity,
        num_convexity_defects,
        max_defect_depth,
        major_axis_length,
        minor_axis_length,
        orientation,
        scaled_huMoments[0],
        centroid_x,
        centroid_y
    ])
    
    return features

def unpackAdvancedFeatures(data, target):
    """Print the features and their corresponding labels."""
    # Check if data and target are correctly populated
    print(f"[INFO] Number of feature vectors: {data.shape[0]}")
    print(f"[INFO] Number of labels: {len(target)}")

    # Print the header (feature names)
    print(f"{'Image #':<10}{'Area':<15}{'Perimeter':<15}{'Aspect Ratio':<15}{'Extent':<15}{'Convex Hull Area':<20}{'Hull Perimeter':<20}{'Solidity':<15}{'Circularity':<15}{'Eccentricity':<15}{'Convexity Defects':<20}{'Max Defect Depth':<20}{'Major Axis Length':<20}{'Minor Axis Length':<20}{'Orientation':<15}{'Hu moments':<15}{'Centroid_x':<15}{'Centroid_y':<15}{'Label'}")

    # Loop through each image's features and print them
    for i, feature_vector in enumerate(data):
        label = target[i]  # Get the label for this image
        
        # Unpack the features
        (area, perimeter, aspect_ratio, extent,
         convex_hull_area, hull_perimeter, solidity, circularity,
         eccentricity, num_convexity_defects, max_defect_depth,
         major_axis_length, minor_axis_length, orientation, scaled_huMoments, centroid_x, centroid_y) = feature_vector
        
        # Print the image index and its features
        print(f"{i:<10}{area:<15.2f}{perimeter:<15.2f}{aspect_ratio:<15.2f}{extent:<15.2f}{convex_hull_area:<20.2f}{hull_perimeter:<20.2f}{solidity:<15.2f}{circularity:<15.2f}{eccentricity:<15.2f}{num_convexity_defects:<20}{max_defect_depth:<20.2f}{major_axis_length:<20.2f}{minor_axis_length:<20.2f}{orientation:<15.2f}{scaled_huMoments:<15.2f}{centroid_x:<15.2f}{centroid_y:<15.2f}{label}")

def excludeBorder(img_BW):
    # Create a border mask to exclude contours touching the image border
    h, w = img_BW.shape
    border_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Draw a border of 1 pixel width
    cv.rectangle(border_mask, (1, 1), (w-2, h-2), 255, 1)
    
    # Find contours again but ignore contours that touch the border
    contours, hierarchy = cv.findContours(img_BW.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    
    for contour in contours:
        # Check if the contour is touching the image border
        if cv.pointPolygonTest(contour, (0, 0), False) < 0:
            valid_contours.append(contour)
    
    # Now work only with valid_contours instead of contours
    if valid_contours:
        contour = max(valid_contours, key=cv.contourArea)
    else:
        raise ValueError("No valid contours found after excluding border.")
    