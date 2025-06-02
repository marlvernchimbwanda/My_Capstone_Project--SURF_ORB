import os
import numpy as np
import cv2
import time
import csv
from lib.visualization import plotting
from lib.visualization.video import play_trip
from tqdm import tqdm


class VisualOdometry():
    def __init__(self, data_dir):
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir, "poses.txt"))
        self.images = self._load_images(os.path.join(data_dir, "image_l"))
        
          # Folder to store match visualizations
        self.matches_dir = os.path.join(data_dir, "surf_orb_matches500")
        os.makedirs(self.matches_dir, exist_ok=True)

        # Initialize SURF for keypoint detection
        self.surf = cv2.xfeatures2d.SURF.create(hessianThreshold=500)
        # Initialize ORB for descriptor extraction
        self.orb = cv2.ORB.create()
        
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    @staticmethod
    def _load_calib(filepath):

        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K (ndarray): Intrinsic parameters
        P (ndarray): Projection matrix
        """

        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def _load_poses(filepath):

        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The GT poses
        """
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        """
        Loads the images

        Parameters
        ----------
        filepath (str): The file path to image dir

        Returns
        -------
        images (list): grayscale images
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    @staticmethod
    def _form_transf(R, t):

        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i):
        """
        This function detect and compute keypoints and descriptors from the i-1'th and i'th image using the surf_orb

        Parameters
        ----------
        i (int): The current frame

        Returns
        -------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        """

        # Detect keypoints using SURF
        kp1 = self.surf.detect(self.images[i - 1], None)
        kp2  = self.surf.detect(self.images[i], None)
        
        # Compute descriptors using ORB
        kp1, des1 = self.orb.compute(self.images[i - 1], kp1)
        kp2, des2 = self.orb.compute(self.images[i], kp2)
        
        matches = self.flann.knnMatch(des1, des2, k=2)
        
        good = [] #list with all good matches
        try:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])

        draw_params = dict(matchColor = -1, # draw matches in green color
                 singlePointColor = None,
                 matchesMask = None, # draw only inliers
                 flags = 2)
        
        img3 = cv2.drawMatches(self.images[i],kp1,self.images[i-1], kp2,good,None, **draw_params)
        cv2.imshow("image", img3)
        cv2.waitKey(750)

        return q1, q2 , len(matches), len(good), kp1, kp2, good

    def get_pose(self, q1, q2):
        """
        Calculates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """
        E, _ = cv2.findEssentialMat(q1, q2, self.K) # the essential matrix derived from the epipolar geometry describes how points in one image corresponds to lines in the other
        # print("Essential Matrix Shape:", E.shape)
        # print("Essential Matrix:", E)

        R, t = self.decomp_essential_mat(E, q1, q2) #after decompose we get rotation and translation
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix

    def decomp_essential_mat(self, E, q1, q2):

        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """
      
        def sum_z_cal_relative_scale(R, t):
            T = self._form_transf(R, t)
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T) #triangualtion to  get 3d world points
            hom_Q2 = np.matmul(T, hom_Q1)
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :] 
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1) /
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
          
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale
        
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]] #4 possible pairs of transformation we can get out 
        
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return [R1, t]


def main():
    root = os.getcwd()
    data_dir = os.path.join(root, 'datasets/KITTI_sequence_7') #try also for 0,1,2,3,4,5,6,7
    vo = VisualOdometry(data_dir)

    play_trip(vo.images)

    gt_path = []
    estimated_path = []
    cur_pose = vo.gt_poses[0]
    cumulative_error = 0.0
    # CSV setup
    csv_path = os.path.join(data_dir, f"{data_dir}_surf_orb_vo_results.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "raw_matches", "inliers", "ms_per_frame", "cumulative_error"])

        for i in tqdm(range(1, len(vo.gt_poses)), unit="frame"):
                start = time.time()

                q1, q2, raw, inliers, kp1, kp2, good = vo.get_matches(i)
                transf = vo.get_pose(q1, q2)
                cur_pose = np.dot(cur_pose, np.linalg.inv(transf))

                gt_pose = vo.gt_poses[i]
                gt_x, gt_z = gt_pose[0, 3], gt_pose[2, 3]
                est_x, est_z = cur_pose[0, 3], cur_pose[2, 3]
                error = np.sqrt((gt_x - est_x) ** 2 + (gt_z - est_z) ** 2)
                cumulative_error += error
                print("\nGround Truth Pose:\n" + str(gt_pose))
                print("\n Current Pose:\n" + str(cur_pose))
                print("The curent pose used x,y: \n" + str(cur_pose[0,3]) + "   " + str(cur_pose[2,3]))

                # Save match visualization
                match_img = cv2.drawMatches(vo.images[i - 1], kp1, vo.images[i], kp2, good, None,
                                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imwrite(os.path.join(vo.matches_dir, f"match_{i:04d}.png"), match_img)

                elapsed_ms = (time.time() - start) * 1000

                writer.writerow([i, raw, inliers, round(elapsed_ms, 2), round(cumulative_error, 4)])

                gt_path.append((gt_x, gt_z))
                estimated_path.append((est_x, est_z))
        

        print(f"[INFO] Results saved to {csv_path}")
        plotting.visualize_paths(gt_path, estimated_path,
                                "SURF-ORB Monocular Visual Odometry",
                                file_out=os.path.basename(data_dir) + "_surf_orb.html")
        

if __name__ == "__main__":
    main()