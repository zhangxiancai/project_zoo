import math
import cv2
import numpy as np

import onnxruntime as ort
#

class inference:


    onnx_model_path='/home/xiancai/face_angle/6DRepNet/results/2022_03_22/_epoch_67_mae9.1463_sim.onnx'
    img_size=(84,84)
    ses = ort.InferenceSession(onnx_model_path)
    # ses.run(None, {'images': img})

    def classify(self, img):

        img= cv2.imread(img) if isinstance(img,str) else img
        img = img[..., ::-1] # to rgb
        img = cv2.resize(img, self.img_size) / 255.0
        # img = cv2.resize(img, (112, 112))
        img = img.transpose(2, 0, 1) # to c*h*w
        img = img[None,...] # to 1*c*h*w
        img = np.ascontiguousarray(img).astype(np.float32)

        # inference
        pre_6d=self.ses.run(None, {'img': img}) # return a list
        pre_6d = pre_6d[0] # 1*6 numpy
        print(pre_6d)

        # 后处理
        R = self.compute_rotation_matrix_from_ortho6d(pre_6d)
        euler = self.compute_euler_angles_from_rotation_matrices(
            R) * 180 / np.pi
        p_pred_deg = euler[:, 0]
        y_pred_deg = euler[:, 1]
        r_pred_deg = euler[:, 2]

        return p_pred_deg,y_pred_deg,r_pred_deg

    # poses batch*6
    # poses
    def compute_rotation_matrix_from_ortho6d(self, poses):
        x_raw = poses[:, 0:3]  # batch*3
        y_raw = poses[:, 3:6]  # batch*3

        x = self.normalize_vector(x_raw)  # batch*3
        z = self.cross_product(x, y_raw)  # batch*3
        z = self.normalize_vector(z)  # batch*3
        y = self.cross_product(z, x)  # batch*3

        x,y,z=x.reshape(1,3,1),y.reshape(1,3,1),z.reshape(1,3,1)
        # matrix = torch.cat((x, y, z), 2)  # batch*3*3
        matrix = np.concatenate((x, y, z), 2)
        return matrix

    # batch*n
    def normalize_vector(self, v):
        v_mag = np.sqrt(np.sum(np.power(v, 2)))
        v_mag=np.max((v_mag,1e-8))
        v = v / v_mag
        return v

    # u, v batch*n
    def cross_product(self, u, v):
        i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        out = np.array([i,j,k]).reshape(1,3)
        return out

    # input batch*4*4 or batch*3*3
    # output torch batch*3 x, y, z in radiant
    # the rotation is in the sequence of x,y,z
    def compute_euler_angles_from_rotation_matrices(self, rotation_matrices):
        # batch = rotation_matrices.shape[0]
        R = rotation_matrices
        sy = np.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
        singular = float(sy < 1e-6)

        x = math.atan2(R[:, 2, 1], R[:, 2, 2])
        y = math.atan2(-R[:, 2, 0], sy)
        z = math.atan2(R[:, 1, 0], R[:, 0, 0])

        xs = math.atan2(-R[:, 1, 2], R[:, 1, 1])
        ys = math.atan2(-R[:, 2, 0], sy)
        zs = R[:, 1, 0] * 0

        out_euler = np.zeros((1,3))
        out_euler[:, 0] = x * (1 - singular) + xs * singular
        out_euler[:, 1] = y * (1 - singular) + ys * singular
        out_euler[:, 2] = z * (1 - singular) + zs * singular

        return out_euler

if __name__=='__main__':

    img='/data1/xiancai/FACE_ANGLE_DATA/test/crop_1.jpg'
    res=inference().classify(img)
    print(res)

