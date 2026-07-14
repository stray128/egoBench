"""Model wrappers. Each exposes the same run() interface so the stage runner
never branches on which model it is calling. Camera-space hand models emit
HandPose(frame=CAMERA); world models emit camera trajectory + world-frame hands.

Heavy models (hamer, mapanything, hawor) are AWS-only, their weights are large
and often non-commercial, installed via scripts/setup_aws.sh, not pip.
"""
