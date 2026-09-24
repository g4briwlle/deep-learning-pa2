"""Module with synthetic ellipse video dataset"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any

class SyntheticTrackerGenerator:
    """
    Generates synthetic ground truth tracks (bounding boxes representing ellipses/pedestrians).
    Simulates objects moving in a 2D space with configurable speed and occlusion.
    """
    def __init__(self, 
                 num_frames: int = 100, 
                 num_objects: int = 5, 
                 img_width: int = 1920, 
                 img_height: int = 1080,
                 max_speed: float = 5.0,
                 occlusion_prob_per_frame: float = 0.02, # Chance of starting an occlusion
                 occlusion_duration: int = 10):          # N frames the object stays occluded
        self.num_frames = num_frames
        self.num_objects = num_objects
        self.img_width = img_width
        self.img_height = img_height
        self.max_speed = max_speed
        self.occlusion_prob_per_frame = occlusion_prob_per_frame
        self.occlusion_duration = occlusion_duration
        
    def generate(self) -> List[Dict[str, Any]]:
        """
        Generates the ground truth trajectories.
        Returns a list of dictionaries containing the MOT17 format fields.
        """
        ground_truth = []
        
        # Initialize object states: [x, y, width, height, vx, vy]
        # x, y represent the top-left corner
        objects = {}
        for obj_id in range(1, self.num_objects + 1):
            width = np.random.uniform(40, 80)
            height = np.random.uniform(100, 180)
            x = np.random.uniform(0, self.img_width - width)
            y = np.random.uniform(0, self.img_height - height)
            vx = np.random.uniform(-self.max_speed, self.max_speed)
            vy = np.random.uniform(-self.max_speed, self.max_speed)
            objects[obj_id] = {
                'x': x, 'y': y, 'w': width, 'h': height, 
                'vx': vx, 'vy': vy, 
                'occlusion_timer': 0 # 0 means visible, > 0 means occluded for that many frames
            }
            
        for frame in range(1, self.num_frames + 1):
            for obj_id, state in objects.items():
                # Update position based on velocity
                state['x'] += state['vx']
                state['y'] += state['vy']
                
                # Bounce off the walls (simple collision detection)
                if state['x'] <= 0 or state['x'] + state['w'] >= self.img_width:
                    state['vx'] *= -1
                    state['x'] = np.clip(state['x'], 0, self.img_width - state['w'])
                if state['y'] <= 0 or state['y'] + state['h'] >= self.img_height:
                    state['vy'] *= -1
                    state['y'] = np.clip(state['y'], 0, self.img_height - state['h'])
                
                # Handle Occlusion Logic
                visibility = 1.0
                
                # If currently occluded, decrement timer
                if state['occlusion_timer'] > 0:
                    state['occlusion_timer'] -= 1
                    visibility = 0.0
                # If visible, roll the dice to see if occlusion starts
                elif np.random.rand() < self.occlusion_prob_per_frame:
                    state['occlusion_timer'] = self.occlusion_duration
                    visibility = 0.0
                
                # Record position if visible (or if you want to track invisible states, change this)
                # The prompt implies the detector doesn't see it when occluded.
                if visibility > 0:
                    ground_truth.append({
                        'frame': frame,
                        'id': obj_id,
                        'bb_left': state['x'],
                        'bb_top': state['y'],
                        'bb_width': state['w'],
                        'bb_height': state['h'],
                        'conf': 1.0,
                        'class': 1,
                        'visibility': visibility
                    })
                    
        return ground_truth



class DetectorSimulator:
    """
    Takes ground truth detections and purposely degrades them to simulate a real detector.
    It drops boxes, adds coordinate noise, and injects false positives.
    """
    def __init__(self, 
                 drop_prob: float = 0.1, 
                 noise_std: float = 5.0, 
                 fp_per_frame: int = 2,
                 img_width: int = 1920,
                 img_height: int = 1080):
        self.drop_prob = drop_prob
        self.noise_std = noise_std
        self.fp_per_frame = fp_per_frame
        self.img_width = img_width
        self.img_height = img_height

    def simulate(self, ground_truth: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applies the degradation pipeline to the ground truth data.
        Returns the simulated detection boxes.
        """
        simulated_detections = []
        
        # 1. Drop True Positives and Add Noise
        for gt_box in ground_truth:
            # Drop p% of the true boxes
            if np.random.rand() < self.drop_prob:
                continue
                
            sim_box = gt_box.copy()
            
            # Add Gaussian noise to coordinates
            sim_box['bb_left'] += np.random.normal(0, self.noise_std)
            sim_box['bb_top'] += np.random.normal(0, self.noise_std)
            sim_box['bb_width'] += np.random.normal(0, self.noise_std)
            sim_box['bb_height'] += np.random.normal(0, self.noise_std)
            
            # Clip values to ensure they stay within image boundaries
            sim_box['bb_left'] = max(0.0, min(sim_box['bb_left'], self.img_width - 1))
            sim_box['bb_top'] = max(0.0, min(sim_box['bb_top'], self.img_height - 1))
            sim_box['bb_width'] = max(1.0, sim_box['bb_width'])
            sim_box['bb_height'] = max(1.0, sim_box['bb_height'])
            
            # IDs are usually unknown (-1) at the detection stage in MOT
            sim_box['id'] = -1 
            simulated_detections.append(sim_box)
            
        # 2. Inject False Positives
        if len(ground_truth) > 0:
            max_frame = max(box['frame'] for box in ground_truth)
            for frame in range(1, max_frame + 1):
                for _ in range(self.fp_per_frame):
                    simulated_detections.append({
                        'frame': frame,
                        'id': -1,
                        'bb_left': np.random.uniform(0, self.img_width - 50),
                        'bb_top': np.random.uniform(0, self.img_height - 100),
                        'bb_width': np.random.uniform(40, 80),
                        'bb_height': np.random.uniform(100, 180),
                        'conf': np.random.uniform(0.1, 0.9), # Lower confidence for FPs
                        'class': 1,
                        'visibility': 1.0
                    })
                    
        # Sort by frame to maintain chronological order
        simulated_detections.sort(key=lambda x: x['frame'])
        return simulated_detections




def plot_trajectory_from_data(data: List[Dict[str, Any]], img_width: int, img_height: int, target_id: int = 1):
    """
    Plots the trajectory of a specific object based on the generated synthetic data.
    Highlights gaps (occlusions) where the object disappeared for N frames.
    """
    # Filter data for the specific target ID and sort chronologically
    track_data = [d for d in data if d['id'] == target_id]
    track_data.sort(key=lambda x: x['frame'])

    if not track_data:
        print(f"Nenhum dado encontrado para o ID {target_id}.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0) # Invert Y axis for image coordinates
    ax.set_title(f'Trajetória do Objeto ID {target_id} com Detecção de Oclusões', fontsize=14)
    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')

    prev_x, prev_y, prev_frame = None, None, None
    
    for item in track_data:
        curr_x = item['bb_left']
        curr_y = item['bb_top']
        w = item['bb_width']
        h = item['bb_height']
        frame = item['frame']
        
        # Center of the bounding box for plotting the line
        center_x = curr_x + w / 2
        center_y = curr_y + h / 2

        # Draw the bounding box
        rect = patches.Rectangle((curr_x, curr_y), w, h, 
                                 linewidth=1, edgecolor='blue', facecolor='none', alpha=0.3)
        ax.add_patch(rect)
        
        # Draw the center point
        ax.plot(center_x, center_y, 'bo', markersize=3)
        
        # Connect points and check for gaps
        if prev_x is not None:
            frame_gap = frame - prev_frame
            if frame_gap > 1:
                # OCLUSÃO DETECTADA: O objeto sumiu por 'frame_gap - 1' quadros
                ax.plot([prev_x, center_x], [prev_y, center_y], 
                        'r--', linewidth=2, label=f'Ocluído por {frame_gap - 1} quadros')
                
                # Anotação no gráfico
                mid_x = (prev_x + center_x) / 2
                mid_y = (prev_y + center_y) / 2
                ax.text(mid_x, mid_y - 20, 
                        f'Some: qd {prev_frame}\nVolta: qd {frame}\n(Gap: {frame_gap-1} qd)', 
                        color='red', fontsize=10, weight='bold', 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            else:
                # Movimento contínuo normal
                ax.plot([prev_x, center_x], [prev_y, center_y], 'b-', alpha=0.6)
                
        # Handle legend (prevent duplicate labels)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        prev_x, prev_y, prev_frame = center_x, center_y, frame

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()





if __name__ == "__main__":
    W, H = 1000, 800

    # Set generator to force occlusion more likely for the plot (higher prob and 15 frames duration)
    generator = SyntheticTrackerGenerator(
        num_frames=60, 
        num_objects=1,   # only on object to cleaner graphic
        img_width=W, 
        img_height=H,
        max_speed=8.0, 
        occlusion_prob_per_frame=0.08, 
        occlusion_duration=15         
    )

    # genarates anotations
    gt_data = generator.generate()
    
    # plot with standard mot data
    plot_trajectory_from_data(gt_data, img_width=W, img_height=H, target_id=1)
    

    # Create an easy scenario: few objects, slow, no occlusion
    generator = SyntheticTrackerGenerator(
        num_frames=50, 
        num_objects=3, 
        max_speed=2.0, 
        occlusion_prob=0.0
    )
    
    # Generate the Ground Truth
    gt_tracks = generator.generate()
    print(f"Generated {len(gt_tracks)} Ground Truth bounding boxes.")
    
    # Degrade the Ground Truth using the Detector Simulator
    simulator = DetectorSimulator(
        drop_prob=0.15,      # Discards 15% of the boxes
        noise_std=3.0,       # Adds minimal noise
        fp_per_frame=1       # Injects 1 false positive per frame
    )
    
    simulated_dets = simulator.simulate(gt_tracks)
    print(f"Generated {len(simulated_dets)} Simulated Detections (with noise, drops, and FPs).")