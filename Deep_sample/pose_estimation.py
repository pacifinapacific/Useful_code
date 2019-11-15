
import torch
import torchvision.models as models 
import torchvision.transforms as transforms
import cv2


"""
COCO_PERSON_KEYPOINT_NAMES = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]

"""


class Pose_Detector():
	def __init__(self,cuda=True):
		self.model=models.detection.keypointrcnn_resnet50_fpn(pretrained=True).eval()
		self.cuda=cuda

		if cuda:
			self.model=self.model.cuda()

	def detect(self,image,image_type="raw"):
		if image_type=="raw":
			image=self.cv2_to_tensor(image)

		self.prediction=self.model(image)

		return self.prediction

	def cv2_to_tensor(self,cv2_image,batch=1):
		img=cv2.cvtColor(cv2_image,cv2.COLOR_BGR2RGB)
		img=img/255.0 #0~1 Normlize
		img=transforms.functional.to_tensor(img)
		if self.cuda:
			img=img.float().cuda()
		if batch==1:
			C,H,W=img.size()[0],img.size()[1],img.size()[2]
			img=img.view(1,C,H,W)

		return img




if __name__ == '__main__':
	pose_detector=Pose_Detector()
	image=cv2.imread("sample.jpg")
	prediction=pose_detector.detect(image)

	keypoints=prediction[0]["keypoints"][0].cpu().detach().numpy()
	for keypoint in keypoints:
		x,y,conf=keypoint[0],keypoint[1],keypoint[2]
		cv2.circle(image,(x,y),5,(255,0,0),-1)
	cv2.imwrite("detection.jpg",image)


