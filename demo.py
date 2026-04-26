"""
GuardNet – Demo / Real-Time Detection Script
Usage:
    python demo.py                        # webcam (default)
    python demo.py --source 0            # webcam index 0
    python demo.py --source video.mp4    # video file
    python demo.py --demo_mode           # no model needed (random predictions)
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.config import MODEL_PATH


def parse_args():
    p = argparse.ArgumentParser(description="GuardNet – real-time violence detection")
    p.add_argument("--source",    default="0",
                   help="Video source: integer webcam index or path to video file")
    p.add_argument("--model",     default=MODEL_PATH,
                   help="Path to trained .h5 model file")
    p.add_argument("--threshold", type=float, default=None,
                   help="Override violence probability threshold (0-1)")
    p.add_argument("--demo_mode", action="store_true",
                   help="Create and use an untrained demo model (no dataset required)")
    return p.parse_args()


def main():
    args = parse_args()

    # Apply threshold override
    if args.threshold is not None:
        import config.config as cfg
        cfg.VIOLENCE_THRESHOLD = float(args.threshold)
        print(f"[INFO] Threshold overridden → {cfg.VIOLENCE_THRESHOLD:.2f}")

    # Demo mode: create an untrained model so the pipeline can be tested
    if args.demo_mode or not os.path.exists(args.model):
        print("[INFO] Demo mode – creating untrained model …")
        from models.guardnet_model import create_demo_model
        create_demo_model(args.model)

    # Resolve source
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    print("\n" + "=" * 60)
    print("  GuardNet – Real-Time Violence Detection")
    print("=" * 60)
    print(f"  Source : {source}")
    print(f"  Model  : {args.model}")
    print("  Press  Q  or  ESC  to quit")
    print("  Press  H  to toggle heatmap overlay")
    print("=" * 60 + "\n")

    from utils.detection_engine import DetectionEngine
    engine = DetectionEngine(model_path=args.model)

    if isinstance(source, int):
        engine.run(source=source, source_name=f"webcam:{source}")
    else:
        if not os.path.exists(source):
            print(f"[ERROR] Video file not found: {source}")
            sys.exit(1)
        engine.run(source=source, source_name=os.path.basename(source))


if __name__ == "__main__":
    main()
