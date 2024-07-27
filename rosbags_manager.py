import rosbag
from pathlib import Path
import numpy as np
import cv2


class RosbagManager():
    def __init__(self, path: Path, name: str):
        self.path = path
        self.name = name
        self.file = Path(self.path, self.name)
        self.bag = None

    def _open_bag(self):
        self.bag = rosbag.Bag(self.file, 'r')
        print(f'Bag {self.name} opened successfully.')

    def _close_bag(self):
        self.bag.close()
        print(f'Bag {self.name} closed successfully.')

    def check_bag(self):
        try:
            self._open_bag()
            print(f'Bag duration: {self.bag.get_end_time() - self.bag.get_start_time()}')
            print(f'Bag messages: {self.bag.get_message_count()}')

            info = self.bag.get_type_and_topic_info()
            topics = info.topics

            info = False
            topic_count = 1

            # print Topics and metadata

            for topic, metadata in topics.items():
                if not info:
                    print(" ROSBAG METADATA")
                    print(f"Message type: {metadata.msg_type}")
                    print(f"Message count: {metadata.message_count}")
                    print(f"Frequency: {metadata.frequency}")
                    info = True
                else:
                    print("----------------------------------------------------------------")
                    print(f"Topic: {topic}")
                    print(f"Metadata: {metadata}")


        except Exception as e:
            print(f'Error while opening rosbag {self.name}: {e}')

        finally:
            self._close_bag()

    def extract_video(self):
        try:
            self._open_bag()

            video_writer = None
            frame_count = 0

            for topic, msg, t in self.bag.read_messages(topics=['/camera/color/image_raw']):
                # Extract frame
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if video_writer is None:
                    # Initialize the video writer with the first frame's properties
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    fps = 30
                    output_path = Path(self.path, self.name.replace('.bag', '.avi'))
                    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (msg.width, msg.height))

                # Write frame to video
                video_writer.write(frame)
                frame_count += 1

                # print(f'Frame {frame_count} extracted.')

            if video_writer is not None:
                video_writer.release()
                print(f'Video extracted successfully to {output_path}')


        except Exception as e:
            print(f'Error while extracting images from rosbag {self.name}: {e}')

        finally:
            self._close_bag()

def main():
    rosbag_path = Path('/media/felipezero/T7 Shield/DATA/thesis/Rosbags/2023_05_05/')
    rosbag_name = 'Test3_12_37_C-R/2023_05_05_12_37_Gera_C-R_Alt.orig.bag'
    bag = RosbagManager(rosbag_path, rosbag_name)
    # bag.check_bag()
    bag.extract_video()


if __name__ == '__main__':
    main()