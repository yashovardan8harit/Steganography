import cv2
import os
import numpy as np

def importance_map(gray):

    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx * sx + sy * sy)

    mag_norm = ( (mag - mag.min()) / (mag.max() - mag.min() + 1e-9) * 255.0 ).astype(np.uint8)
    return mag_norm

def load_encrypted_message(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read().strip()

def embed_message_in_frame(frame, message_chunk):

    frame_array = np.array(frame)
    h, w, _ = frame_array.shape

    gray = cv2.cvtColor(frame_array, cv2.COLOR_BGR2GRAY)
    imp = importance_map(gray)

    coords = np.dstack(
        np.unravel_index(
            np.argsort(-imp.ravel()),
            (h, w)
        )
    )[0]

    message_bytes = message_chunk.encode("utf-8")
    total_bits = len(message_bytes) * 8

    if total_bits > len(coords):
        total_bits = len(coords)

    bit_index = 0

    for byte in message_bytes:
        for bit_pos in range(7, -1, -1):
            if bit_index >= total_bits:
                break

            r, c = coords[bit_index]
            bit = (byte >> bit_pos) & 0x01

            frame_array[r, c, 0] = (frame_array[r, c, 0] & 0xFE) | bit

            bit_index += 1
        if bit_index >= total_bits:
            break

    return frame_array

def embed_message_in_video(frames, message, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    chars_per_frame = (frames[0].shape[0] * frames[0].shape[1]) // 8
    total_frames_needed = (len(message) + chars_per_frame - 1) // chars_per_frame

    if total_frames_needed > len(frames):
        raise ValueError("Not enough frames to embed the entire message.")

    frame_count = 0

    for frame in frames:
        if frame_count * chars_per_frame < len(message):
            chunk = message[frame_count * chars_per_frame : (frame_count + 1) * chars_per_frame]
            modified_frame = embed_message_in_frame(frame, chunk)
            modified_frame_path = os.path.join(output_folder, f"modified_frame_{frame_count:04d}.png")
            cv2.imwrite(modified_frame_path, modified_frame)
            print(f"Embedded chunk into frame {frame_count} and saved as {modified_frame_path}.")
        else:
            modified_frame_path = os.path.join(output_folder, f"modified_frame_{frame_count:04d}.png")
            cv2.imwrite(modified_frame_path, frame)
            print(f"Saved remaining frame {frame_count} without modification as {modified_frame_path}.")

        frame_count += 1

    print(f"Message embedding complete. Total frames processed: {frame_count}")

def combine_frames_to_video(frames_folder, output_video_path, frame_rate=30):
    frame_files = sorted([
        f for f in os.listdir(frames_folder) 
        if f.lower().endswith('.png')
    ])

    if not frame_files:
        raise ValueError("No frames found to create the video.")

    first_frame_path = os.path.join(frames_folder, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        raise ValueError(f"Unable to read the first frame: {first_frame_path}")

    height, width, layers = first_frame.shape
    frame_size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, frame_size)

    print(f"Combining frames into video: {output_video_path}")
    print(f"Frame size: {frame_size}, Frame rate: {frame_rate} FPS")

    for filename in frame_files:
        frame_path = os.path.join(frames_folder, filename)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Unable to read frame {filename}. Skipping.")
            continue

        if frame.shape[1] != width or frame.shape[0] != height:
            print(f"Warning: Frame {filename} has a different size. Resizing to match the first frame.")
            frame = cv2.resize(frame, frame_size)

        out.write(frame)
        print(f"Added frame {filename} to video.")

    out.release()
    print(f"Video creation complete: {output_video_path}")


if __name__ == "__main__":
    encrypted_message_filename = r"D:\Steganography\encrypted_message.txt"
    output_frames_folder = r"D:\Steganography\output_frames_with_message"
    original_frames_folder = r"D:\Steganography\output_frames"
    output_video_path = r"D:\Steganography\output_video_with_message.mp4"
    frame_rate = 30

    encrypted_message = load_encrypted_message(encrypted_message_filename)
    print(f"Encrypted message loaded. Length: {len(encrypted_message)} characters.")

    frames = []
    for filename in sorted(os.listdir(original_frames_folder)):
        if filename.lower().endswith('.jpg'):
            frame_path = os.path.join(original_frames_folder, filename)
            frame = cv2.imread(frame_path)
            if frame is not None:
                frames.append(frame)
            else:
                print(f"Warning: Unable to read frame {filename}.")

    if not frames:
        raise ValueError("No frames found to embed the message.")

    print(f"Total frames loaded: {len(frames)}")

    for idx, frame in enumerate(frames):
        print(f"Frame {idx}: Shape {frame.shape}")

    embed_message_in_video(frames, encrypted_message, output_frames_folder)

    combine_frames_to_video(
        frames_folder=output_frames_folder,
        output_video_path=output_video_path,
        frame_rate=frame_rate
    )