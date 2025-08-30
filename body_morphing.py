import cv2
import os
import math
import numpy as np
import mediapipe as mp
from moviepy.editor import VideoFileClip, concatenate_videoclips


# ---------------------------
# HANDS ON WAIST CHECK FUNCTION
# ---------------------------
def hands_on_waist_checking(frame):
    """
    Checks if hands are on the waist in the given frame.
    Returns True if detected, otherwise False.
    """
    mp_pose = mp.solutions.pose

    def angle_between(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            return 0
        return math.degrees(math.acos(dot / norm))

    def get_point(lm):
        return np.array([lm.x, lm.y])

    # Convert to RGB and run pose detection
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_elbow_angle = angle_between(
                get_point(landmarks[11]), get_point(landmarks[13]), get_point(landmarks[15])
            )
            right_elbow_angle = angle_between(
                get_point(landmarks[12]), get_point(landmarks[14]), get_point(landmarks[16])
            )

            if 45 < left_elbow_angle < 100 and 45 < right_elbow_angle < 100:
                return True

    return False


# ---------------------------
# PART 1: Record until hands on waist
# ---------------------------
def videopart1(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_video = "trimmed_output.mp4"
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    print("Recording started... Stop when hands on waist.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if hands_on_waist_checking(frame):
            cv2.imwrite("gesture_1.jpg", frame)
            print("Hands detected -> gesture_1.jpg saved")
            break

        out.write(frame)
        cv2.imshow("Video Part 1", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Trimmed video saved at {output_video}")
    return output_video


# ---------------------------
# SUBTRACT REMAINING VIDEO
# ---------------------------
def videosubtraction(e,original_path, trimmed_path, output_path):
    original = VideoFileClip(original_path)
    trimmed = VideoFileClip(trimmed_path)
    end_time = trimmed.duration

    remaining = original.subclip(end_time + e, original.duration)  # starts after trimmed
    remaining.write_videofile(output_path, codec="libx264", audio_codec="aac")

    original.close()
    trimmed.close()
    remaining.close()
    return output_path


# ---------------------------
# PART 2: Record again after hands on waist
# ---------------------------
def videopart2(video_path):
    cap = cv2.VideoCapture(video_path)
    #frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_video = "final_recording.mp4"
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    recording = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not recording and hands_on_waist_checking(frame):
            recording = True
            cv2.imwrite("gesture_2.jpg", frame)
            print("Hands detected again -> gesture_2.jpg saved")

        if recording:
            out.write(frame)

        cv2.imshow("Video Part 2", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Final recording saved at {output_video}")
    return output_video


# ---------------------------
# MORPHING FUNCTION
# ---------------------------
def morphing(img1_path, img2_path, output="morph_output.mp4",
             duration=2, frame_rate=30):

    # -----------------------------
    # Helper functions
    # -----------------------------
    def apply_affine_transform(src, srcTri, dstTri, size):
        warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
        return cv2.warpAffine(src, warpMat, (size[0], size[1]),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    def morph_triangle(img1, img2, img, t1, t2, t, alpha):
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))
        r = cv2.boundingRect(np.float32([t]))

        t1Rect, t2Rect, tRect = [], [], []
        for i in range(3):
            tRect.append((t[i][0] - r[0], t[i][1] - r[1]))
            t1Rect.append((t1[i][0] - r1[0], t1[i][1] - r1[1]))
            t2Rect.append((t2[i][0] - r2[0], t2[i][1] - r2[1]))

        mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

        img1Rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
        img2Rect = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]

        size = (r[2], r[3])
        warpImage1 = apply_affine_transform(img1Rect, t1Rect, tRect, size)
        warpImage2 = apply_affine_transform(img2Rect, t2Rect, tRect, size)

        imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2
        img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = \
            img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * (1 - mask) + imgRect * mask

    def align_images(img1, img2, max_width=600):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        new_h, new_w = min(h1, h2), min(w1, w2)
        img1_resized = cv2.resize(img1, (new_w, new_h))
        img2_resized = cv2.resize(img2, (new_w, new_h))

        if new_w > max_width:
            scale = max_width / new_w
            new_size = (int(new_w * scale), int(new_h * scale))
            img1_resized = cv2.resize(img1_resized, new_size)
            img2_resized = cv2.resize(img2_resized, new_size)

        return img1_resized, img2_resized

    def get_body_landmarks(img, require_body=True):
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        points = []
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                x = int(lm.x * img.shape[1])
                y = int(lm.y * img.shape[0])
                points.append((x, y))
        elif require_body:
            raise ValueError("❌ Body landmarks not detected in one image.")

        h, w = img.shape[:2]
        points.extend([
            (0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1),
            (w // 2, 0), (w - 1, h // 2), (w // 2, h - 1), (0, h // 2)
        ])
        pose.close()
        return np.array(points)

    def make_delaunay(w, h, points):
        subdiv = cv2.Subdiv2D((0, 0, w, h))
        pts = [(int(p[0]), int(p[1])) for p in points]
        pts_dict = {pt: i for i, pt in enumerate(pts)}
        for p in pts:
            subdiv.insert(p)

        tri_indices = []
        for t in subdiv.getTriangleList():
            pts_tri = [(int(t[0]), int(t[1])),
                       (int(t[2]), int(t[3])),
                       (int(t[4]), int(t[5]))]
            rect = (0, 0, w, h)
            if all(0 <= pt[0] < w and 0 <= pt[1] < h for pt in pts_tri):
                try:
                    ind = [pts_dict[pt] for pt in pts_tri]
                    tri_indices.append(tuple(ind))
                except:
                    continue
        return tri_indices

    def generate_morph_sequence(img1, img2, points1, points2, tri_list, size, output):
        num_frames = int(duration * frame_rate)
        out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'),
                              frame_rate, (size[1], size[0]))

        for i in range(num_frames):
            alpha = i / (num_frames - 1)
            points = [(1 - alpha) * p1 + alpha * p2 for p1, p2 in zip(points1, points2)]
            morphed_img = np.zeros(img1.shape, dtype=np.float32)

            for tri in tri_list:
                t1 = [points1[tri[0]], points1[tri[1]], points1[tri[2]]]
                t2 = [points2[tri[0]], points2[tri[1]], points2[tri[2]]]
                t = [points[tri[0]], points[tri[1]], points[tri[2]]]
                morph_triangle(img1, img2, morphed_img, t1, t2, t, alpha)

            out.write(cv2.convertScaleAbs(morphed_img))
        out.release()
    img1, img2 = cv2.imread(img1_path), cv2.imread(img2_path)
    if img1 is None or img2 is None:
        raise FileNotFoundError(" One of the input images could not be loaded.")

    img1, img2 = align_images(img1, img2)
    size = (img1.shape[0], img1.shape[1])

    points1, points2 = get_body_landmarks(img1), get_body_landmarks(img2)
    tri_list = make_delaunay(size[1], size[0], (points1 + points2) / 2)

    generate_morph_sequence(np.float32(img1), np.float32(img2),
                            points1, points2, tri_list, size, output)

    print(f"Body morph video generated: {output}")
    return output
# CONCATENATE FUNCTION
# ---------------------------
def concat_videos(video_list, output="merged_result.mp4"):
    clips = [VideoFileClip(v) for v in video_list]
    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(output, codec="libx264", audio_codec="aac")
    print(f"Final merged video saved at {output}")


# ---------------------------
# MAIN PIPELINE
# ---------------------------
if __name__ == "__main__":
    input_video = input("enter vdo path")  #i want to use upload vdo option
    e=int(input("enter the gap interval time "))
    trimmed = videopart1(input_video)
    remaining = videosubtraction(e,input_video, trimmed, "final_recording_matched.mp4")
    recorded = videopart2(remaining)
    morph = morphing("gesture_1.jpg", "gesture_2.jpg")
    concat_videos([trimmed, morph, recorded], "merged_result.mp4")
    import os

def delete_files(*file_paths):
    """
    Delete specific files given as arguments.
    Example: delete_files("video.jpg", "photo.jpg")
    """
    for file in file_paths:
        if os.path.exists(file):
            os.remove(file)
            print(f"Deleted: {file}")
        else:
            print(f"File not found: {file}")

# Example usage:
delete_files("gesture_1.jpg", "gesture_2.jpg")
    
def delete_videos(*video_paths):
    """
    Delete specific MP4 video files.
    Example: delete_videos("test.mp4", "output.mp4")
    """
    for file in video_paths:
        if os.path.exists(file):
            os.remove(file)
            print(f"Deleted: {file}")
        else:
            print(f"File not found: {file}")

# Example usage:
delete_videos("part_1.mp4", "part_2.mp4","testing.mp4")
print(" Pipeline complete!")
