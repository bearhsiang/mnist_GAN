import matplotlib.pyplot as plt

def save_img(img, name):
    img = img*0.3081 + 0.1307
    img = img.squeeze()
    plt.imshow(img, cmap='gray')
    plt.savefig(name)