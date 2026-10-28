"""Save 2D and 3D trajectory JPGs. Requires numpy and matplotlib."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('csv', type=Path)
parser.add_argument('--output-dir', type=Path, default=Path('results'))
parser.add_argument('--note', default='Open-loop IMU scale; not an accuracy-validated trajectory')
args = parser.parse_args()
rows = np.atleast_1d(np.genfromtxt(args.csv, delimiter=',', names=True))
xyz = np.column_stack([rows[name] for name in ('x', 'y', 'z')])
valid = np.isfinite(xyz).all(axis=1)
if not valid.any():
    raise SystemExit('No finite positions to plot')
points = xyz[valid]
args.output_dir.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
subtitle = f'{args.note}\n{valid.sum()} finite positions / {len(rows)} frames; gaps are failed visual poses'
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
for ax, (a,b) in zip(axes, [(0,1),(0,2),(1,2)]):
    ax.plot(xyz[:,a], xyz[:,b], color='#276fa8', linewidth=1, marker='.', markersize=3)
    ax.scatter(points[0,a], points[0,b], color='#238443', s=65, label='Start', zorder=5)
    ax.scatter(points[-1,a], points[-1,b], color='#d7301f', marker='X', s=65, label='Last valid', zorder=5)
    ax.set(xlabel=f'{"XYZ"[a]} (m)', ylabel=f'{"XYZ"[b]} (m)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=.25)
axes[0].legend()
fig.suptitle('Camera trajectory — 2D projections\n'+subtitle, fontsize=12)
fig.tight_layout(rect=(0,0,1,.84))
fig.savefig(args.output_dir/'trajectory_2d.jpg', dpi=180, facecolor='white')
plt.close(fig)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(xyz[:,0],xyz[:,1],xyz[:,2], color='#276fa8', linewidth=1, marker='.', markersize=3)
ax.scatter(*points[0], color='#238443', s=65, label='Start')
ax.scatter(*points[-1], color='#d7301f', marker='X', s=65, label='Last valid')
center=(points.max(axis=0)+points.min(axis=0))/2
radius=max(float(np.ptp(points,axis=0).max())/2, 0.001)*1.08
ax.set_xlim(center[0]-radius,center[0]+radius)
ax.set_ylim(center[1]-radius,center[1]+radius)
ax.set_zlim(center[2]-radius,center[2]+radius)
ax.set_box_aspect((1,1,1))
ax.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)')
ax.legend()
ax.view_init(elev=24,azim=-55)
fig.suptitle('Camera trajectory — 3D\n'+subtitle, fontsize=12)
fig.subplots_adjust(top=.82,bottom=.08)
fig.savefig(args.output_dir/'trajectory_3d.jpg',dpi=180,facecolor='white')
plt.close(fig)
print(f'Saved plots to {args.output_dir}; {valid.sum()} valid, {(~valid).sum()} invalid rows')
