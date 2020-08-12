import cv2
import matplotlib.pyplot as plt # used for data visualization
# plots query image different and then subplot of 3x3 total 9 images in a single plot
def plot_query_retrieval(img_query, img_retrieval):
    plt.figure()
    plt.imshow(cv2.cvtColor(img_query, cv2.COLOR_BGR2RGB))
    plt.show()
    for i, img in enumerate(img_retrieval[0]):
        plt.subplot(3,3,i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
