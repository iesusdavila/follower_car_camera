#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import os

class FollowerCar(Node):
    def __init__(self):
        super().__init__('follower_car')
        
        # Suscribirse al tópico de imagen de la cámara
        self.image_subscription = self.create_subscription(
            Image,
            'tb2/camera/image_raw',
            self.image_callback,
            10
        )
        
        # Publicador para enviar comandos de velocidad
        self.cmd_vel_publisher = self.create_publisher(Twist, '/tb2/cmd_vel', 10)
        
        # Inicializar OpenCV y YOLO
        self.bridge = CvBridge()

        self.net = cv2.dnn.readNet(
            '/home/rov-robot/yolo_weights/yolov4.weights',
            '/home/rov-robot/yolo_weights/yolov4.cfg'
        )
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.target_class = "car"  # Clase objetivo para seguimiento
        
        # Parámetros de control de velocidad
        self.follow_distance = 1.0  # Distancia deseada en metros (ajustable)
        self.linear_speed = 0.1
        self.angular_speed_factor = 0.0005
        
        self.get_logger().info("FollowerCar node initialized and ready to follow vehicles")

    def image_callback(self, msg):
        # Convertir el mensaje de imagen a formato OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Preprocesar la imagen para YOLO
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (608, 608), swapRB=True, crop=False)
        self.net.setInput(blob)
        detections = self.net.forward(self.output_layers)

        # Variables para almacenar la detección más cercana del objetivo
        target_box = None
        max_confidence = 0.5  # Umbral mínimo de confianza

        # Procesar las detecciones
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > max_confidence:
                    center_x = int(obj[0] * frame.shape[1])
                    center_y = int(obj[1] * frame.shape[0])
                    width = int(obj[2] * frame.shape[1])
                    height = int(obj[3] * frame.shape[0])
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    
                    # Asignar el cuadro delimitador del objeto detectado
                    target_box = [x, y, width, height]
                    max_confidence = confidence

        # Realizar seguimiento del vehículo si se ha detectado uno
        if target_box:
            x, y, w, h = target_box
            self.get_logger().info(f"Detected car with confidence {max_confidence:.2f} at [{x}, {y}, {w}, {h}]")

            # Calcular el movimiento de seguimiento
            center_offset = (frame.shape[1] / 2) - (x + w / 2)
            distance_error = self.follow_distance - (w / frame.shape[1])

            # Comandos de velocidad
            twist = Twist()
            twist.linear.x = self.linear_speed if distance_error > 0 else 0.0
            twist.angular.z = center_offset * self.angular_speed_factor

            self.cmd_vel_publisher.publish(twist)

            # Dibujar el cuadro delimitador
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            self.get_logger().info("No car detected; stopping.")
            # Detener si no se detecta el objetivo
            self.cmd_vel_publisher.publish(Twist())
        
        # Mostrar la imagen con detecciones
        cv2.imshow("FollowerCar - Vehicle Detection", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = FollowerCar()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down FollowerCar node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
