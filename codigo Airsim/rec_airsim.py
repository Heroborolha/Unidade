import airsim
import cv2
import numpy as np

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()
client.takeoffAsync().join()
client.takeoffAsync().join()
client.takeoffAsync().join()
client.takeoffAsync().join()
client.takeoffAsync().join()
client.takeoffAsync().join()
client.takeoffAsync().join()

def transform_input(responses):
    img1d = np.frombuffer(responses.image_data_uint8, dtype=np.uint8)
    img_rgba = img1d.reshape(responses.height, responses.width, 3)

    from PIL import Image

    image = Image.fromarray(img_rgba)

    im_final = np.array(image.convert('RGB'))

    return im_final

# Load Yolo
net = cv2.dnn.readNetFromDarknet("yolov4-custom.cfg", "yolov5.weights")
classes = []
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

while(True):
    # Loading image
    img = client.simGetImages([airsim.ImageRequest("3", airsim.ImageType.Scene, False, False)])[0]
    img = transform_input(img)
    img = cv2.resize(img, None, fx=3, fy=3)
    altura, largura, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                centro_x = int(detection[0] * largura)
                centro_y = int(detection[1] * altura)
                w = int(detection[2] * largura)
                h = int(detection[3] * altura)
                # Rectangle coordinates
                x = int(centro_x - w / 2)
                y = int(centro_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    fonte = cv2.FONT_HERSHEY_PLAIN

    ObjectWidthPlusHeight = []
    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y + 30), fonte, 1, (0, 255, 0), 3)
            imagem_centro = largura / 2
            desvio = centro_x - imagem_centro
            voar = desvio / imagem_centro
            velocidade = 5
            ganho_de_voo = 0.5
            yaw_rate = ganho_de_voo * voar
            client.rotateByYawRateAsync(yaw_rate=yaw_rate, duration=0.1)
            client.moveByVelocityAsync(velocidade, 0, 0, duration=0.1)
            

    cv2.imshow('img',img)
    key = cv2.waitKey(1)

    