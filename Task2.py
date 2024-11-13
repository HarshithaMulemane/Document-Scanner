import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'task2.jpeg'
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print("Failed to load image. Please check the file path and format.")
else:
    def scan_document(image):
        # Make a copy of the image for processing
        original = image.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection using Canny
        edged = cv2.Canny(blurred, 50, 150)
        
        # Find contours in the edged image
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area, and assume the largest 4-point contour is the ID card
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        document_contour = None

        for contour in contours:
            # Approximate the contour to a polygon
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            # If the approximated polygon has four vertices, it’s likely our document
            if len(approx) == 4:
                document_contour = approx
                break

        if document_contour is None:
            raise ValueError("Could not find a document in the image.")

        # Order the points of the document contour to get a consistent transformation
        def order_points(pts):
            rect = np.zeros((4, 2), dtype="float32")
            
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]  # Top-left
            rect[2] = pts[np.argmax(s)]  # Bottom-right
            
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]  # Top-right
            rect[3] = pts[np.argmax(diff)]  # Bottom-left
            
            return rect

        # Transform the contour points to a rectangle
        doc_pts = document_contour.reshape(4, 2)
        rect = order_points(doc_pts)
        (tl, tr, br, bl) = rect

        # Calculate the width and height of the new "scanned" image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Destination points for the top-down view of the document
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # Compute the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        scanned = cv2.warpPerspective(original, M, (maxWidth, maxHeight))

        return scanned

    # Process the image to get the scanned result
    scanned_result = scan_document(image)

    # Display the original and scanned document side by side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Scanned ID Card")
    plt.imshow(cv2.cvtColor(scanned_result, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
