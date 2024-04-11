import topologyLib as topo
import generate_symImg as symimg
import cv2

image_path = "DatasetOrg/MNIST/mnist_png/train/8/17.png"

image = symimg.load_image(image_path)

symmetric_image = symimg.make_symmetric(image)

# cv2.imshow('Symmetric Image', symmetric_image)

# Wait for a key press and then close the displayed windows
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print("Image loaded numpy array: ")
print(symmetric_image)

# Now test methods from TopologyLib

ccs, cycles = topo.adj2pers(symmetric_image)

print()
print("Connected Components: ")
print(ccs)
print()
print("Cycles: ")
print(cycles)

# test peer to vec

ret = topo.pers2vec(ccs, cycles, 119, 7021)

print()
print("Pers2Vec: ")
print(ret)