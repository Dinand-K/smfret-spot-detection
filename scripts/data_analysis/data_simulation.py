#IMPORTS
import numpy as np
import tifffile as tiff
import sys

#Snakemake inputs and outputs
path_tiff    = snakemake.output.tiff
path_coords  = snakemake.output.coords
path_classes = snakemake.output.classes
sim_config   = snakemake.params.sim_settings
use_real_data_boolean = snakemake.params.use_real_data
rep = snakemake.wildcards.rep


###EXPERIMENTAL VIDEO PATH###


if use_real_data_boolean:
    print("Data simulation script called to use real data")
    
    src_tiff = "realdata/vid.tiff"
    src_coords = "realdata/coord.npy"
    src_classes = "realdata/class.npy"

    video = tiff.imread(src_tiff)
    video = video[:, [1, 0], :, :]
    
    #same format as simulated videos (float32, ImageJ TCYX)
    tiff.imwrite(
    path_tiff, 
    video.astype(np.float32), 
    imagej=True, 
    metadata={'axes': 'TCYX'}
    )

    np.save(path_coords, np.load(src_coords))
    np.save(path_classes, np.load(src_classes))

    print(f"Real data processed and saved to {path_tiff}!")

    sys.exit(0)


###DATA SIMULATOR FUNCTIONS###


print(f"Starting data simulation #{rep}...")

def generate_valid_simulation_tracks(n_stat, n_rw, n_agg,
                                     n_frames, img_size, margin, min_distance,
                                     rw_step_size, rw_min_life, rw_max_life, rng):
    total_particles = n_stat + n_rw + n_agg
    
    if total_particles == 0:
        return np.empty((n_frames,0,2)), np.empty((n_frames,0,2)), np.array([]), np.array([]), np.empty((n_frames,0,2))

    valid_tracks = []
    valid_gt_positions = []
    types = [] # 0=Stat, 1=RW, 2=Agg
    rw_starts, rw_ends = [], []  # #First and last frame of each accepted random walker

    type_queue = ([0]*n_stat) + ([1]*n_rw) + ([2]*n_agg)
    rng.shuffle(type_queue)

    low = float(margin)
    high = float(img_size - margin)
    max_consecutive_fails = int(img_size * 10)
    consecutive_fails = 0

    #places particles one by one, rejecting any closer than min_distance to an accepted one (mobile contaminants checked at their mean position)
    for p_type in type_queue:
        placed = False
        while not placed:
            if consecutive_fails > max_consecutive_fails:
                raise RuntimeError("Could not place all particles, lower amount of particles.")

            #Set up random starting point and track for stat and agg
            start = rng.uniform(low, high, size=2)
            track = np.full((n_frames, 2), np.nan)
            gt_point = start.copy() #Default for Stat/Agg

            if p_type == 0 or p_type == 2: # Stationary / Aggregate
                track[:] = start
            
            #For Random walk we need to simulate it to find average position
            elif p_type == 1: # Random Walk
                t_start = rng.integers(0, n_frames)
                life = rng.integers(rw_min_life, min(rw_max_life, n_frames) + 1) #if max life is greater than frames needs capping
                t_end = min(t_start + life - 1, n_frames - 1) #cut off at the end
                L = t_end - t_start + 1
                
                track[t_start] = start
                if L > 1:
                    steps = rng.uniform(-rw_step_size, rw_step_size, size=(L - 1, 2))
                    traj = np.cumsum(steps, axis=0) + start
                    traj = np.clip(traj, 0, img_size - 1)
                    track[t_start+1 : t_start+L] = traj
                
                #check mean position for safety
                valid_segment = track[t_start : t_start+L]
                gt_point = np.mean(valid_segment, axis=0)

            #Distance Check
            if len(valid_gt_positions) > 0:
                existing_arr = np.array(valid_gt_positions)
                dists = np.linalg.norm(existing_arr - gt_point, axis=1)#distance candidate to all accepted ground truth positions 
                
                if np.any(dists < min_distance):
                    consecutive_fails += 1
                    continue

            #Success
            valid_tracks.append(track)
            valid_gt_positions.append(gt_point)
            types.append(p_type)

            #Also save the track's start and end times 
            if p_type == 1:
                rw_starts.append(t_start)
                rw_ends.append(t_end)
            consecutive_fails = 0

            placed = True

    # Reshape to (Time, Particles, 2)
    all_tracks = np.array(valid_tracks) 
    all_tracks = np.moveaxis(all_tracks, 0, 1) 
    types = np.array(types)

    # Slice back into groups
    stat_tracks = all_tracks[:, types==0, :]
    rw_tracks   = all_tracks[:, types==1, :]
    agg_tracks  = all_tracks[:, types==2, :]

    return stat_tracks, rw_tracks, np.array(rw_starts, dtype=int), np.array(rw_ends, dtype=int), agg_tracks

def generate_stationary_intensities_hmm(n_frames, num_particles, rng,
                                        pi=None, trans_matrix=None,
                                        fret_means=(0.2, 0.8), fret_stds=(0.05, 0.05),
                                        min_total_brightness=1.0, max_total_brightness=3.0):
    if num_particles == 0: return np.empty((n_frames, 0, 2), dtype=float)

    trans_matrix = np.asarray(trans_matrix, dtype=float)
    fret_means = np.asarray(fret_means, dtype=float)
    fret_stds = np.asarray(fret_stds, dtype=float)
    
    total_brightness_list = rng.uniform(min_total_brightness, max_total_brightness, size=num_particles) #Array of total brightness (D+A) for each particle
    intensities = np.empty((n_frames, num_particles, 2), dtype=float)#emty array to hold intensities for all particles over time
    cum_trans = np.cumsum(trans_matrix, axis=1)#cumulative for easy sampling

    for i in range(num_particles):

        #state generation
        states = np.empty(n_frames, dtype=int)

        #initial state based on pi
        r0 = rng.random()
        state0 = 0 if r0 < pi[0] else 1
        states[0] = state0

        #using cumulative trans matrix to get next states
        for t in range(1, n_frames):
            r = rng.random()
            prev = states[t-1]
            states[t] = 0 if r < cum_trans[prev, 0] else 1

        #FRET efficiency over time, stochastic to mirror molecule movement
        meansovertime = fret_means[states]
        stdsovertime = fret_stds[states]
        efficiency_trace = np.clip(rng.normal(loc=meansovertime, scale=stdsovertime, size=n_frames), 0.0, 1.0)
        
        #Intensity Calculation
        total_brightness = total_brightness_list[i]
        intensities[:, i, 0] = total_brightness * (1.0 - efficiency_trace) # Donor
        intensities[:, i, 1] = total_brightness * efficiency_trace         # Acceptor

    return intensities

def generate_moving_intensities(n_frames, num_particles, t_starts, t_ends, rng, min_brightness=0.5, max_brightness=2.0, fade_len=1):
    if num_particles == 0: return np.empty((n_frames, 0, 2), dtype=float)

    intensities = np.full((n_frames, num_particles, 2), np.nan, dtype=float)
    brightnesses = rng.uniform(min_brightness, max_brightness, num_particles)
    
    for i in range(num_particles):
        s, e = int(t_starts[i]), int(t_ends[i])
        L = e - s + 1
        fade = min(fade_len, L // 2)
        
        #Fade in and fade out, normalized
        rise = np.linspace(0, 1, fade)
        flat = np.ones(max(0, L - 2 * fade))
        fall = np.linspace(1, 0, fade)
        full_curve = np.concatenate([rise, flat, fall])

        #Then multiplied by brightness
        trace = brightnesses[i] * full_curve
        intensities[s : e + 1, i, :] = np.stack([trace, trace], axis=1)
        
    return intensities

def generate_aggregate_intensities(n_frames, num_aggregates, aggpair_count_range,
                                   aggpair_brightness_mean, aggpair_brightness_cv, aggpair_pi,
                                   aggpair_fret_means, aggpair_fret_stds, aggpair_trans_matrix,
                                   jitter_std, rng):
    #aggpair_(somethingsomething) = one FRET pair inside an aggregate. Each pair is generated independently (using the earlier function) and added together to form the aggregate's intensity trace.
    #Each pair brightness is normal(mean, cv * mean) so aggpair_brightness_cv is relative SD (SD = cv * mean)

    n_aggpairs = rng.integers(aggpair_count_range[0], aggpair_count_range[1] + 1, size=num_aggregates) 

    agg_intensities = np.zeros((n_frames, num_aggregates, 2), dtype=float)

    for i in range(num_aggregates):
        N = int(n_aggpairs[i])
        agg_trace_accum = np.zeros((n_frames, 2), dtype=float)

        for j in range(N):
            #normal distr aggpair brightness using cv as mentioned above
            pair_brightness = max(0.1, rng.normal(loc=aggpair_brightness_mean, scale=aggpair_brightness_cv * aggpair_brightness_mean))

            single_trace = generate_stationary_intensities_hmm(
                n_frames=n_frames, num_particles=1, pi=aggpair_pi, trans_matrix=aggpair_trans_matrix,
                fret_means=aggpair_fret_means, fret_stds=aggpair_fret_stds,
                min_total_brightness=pair_brightness, max_total_brightness=pair_brightness,
                rng=rng  # same generator: each pair draws the next random numbers, so pairs switch FRET independently
            )
            agg_trace_accum += single_trace[:, 0, :]
    
        # Apply global jitter
        jitter = rng.normal(loc=1, scale=jitter_std, size=(n_frames,2))
        agg_intensities[:, i, :] = np.clip(agg_trace_accum * jitter, 0.0, None)
        
    return agg_intensities

def render_frames(stationary_positions, rw_positions, agg_positions, 
                  stationary_intensities, rw_intensities, agg_intensities, 
                  num_stationary, num_randomwalk, num_aggregates, 
                  n_frames, img_size, gauss_sigma, rng,
                  photon_scale=1000, read_noise_sigma=3.0, background_level=0.01):

    frames = []

    # Pre-calculate Gaussian kernel
    radius = int(np.ceil(4 * gauss_sigma)) #beyond 4 sigma is only around 0.03% of the peak
    offsets = np.arange(-radius, radius + 1)  # [-6,-5,...,0,...,5,6] for radius=6
    dy_grid, dx_grid = np.meshgrid(offsets, offsets, indexing='ij')  # 13x13 grids of row and column offsets
    
    gaussian_norm_factor = 1.0 / (2 * np.pi * gauss_sigma**2)

    def add_particles(frame, positions, intensities):
        valid_mask = ~np.isnan(positions).any(axis=1)
        valid_pos = positions[valid_mask]
        valid_int = intensities[valid_mask]

        for (x, y), (i_donor, i_acceptor) in zip(valid_pos, valid_int):
            cx, cy = int(np.round(x)), int(np.round(y))
            shift_x, shift_y = cx - x, cy - y
            
            # Sub-pixel Weights
            dist_sq = (dx_grid + shift_x)**2 + (dy_grid + shift_y)**2 # distance squared from the center of the Gaussian using the sub-pixel shift and the precomputed grid of offsets dx ad dy
            weights = np.exp(-dist_sq / (2 * gauss_sigma**2))
            weights *= gaussian_norm_factor
            
            #Add light to the padded frame (radius px border on every side) so the patch always fits,
            frame[cy:cy + 2 * radius + 1, cx:cx + 2 * radius + 1, 0] += i_donor * weights
            frame[cy:cy + 2 * radius + 1, cx:cx + 2 * radius + 1, 1] += i_acceptor * weights

    for t in range(n_frames):
        frame = np.zeros((img_size + 2 * radius, img_size + 2 * radius, 2), dtype=np.float32)  # padded by radius sigma kernel, after plotting the coordinates the final video is corpped again

        add_particles(frame, stationary_positions[t], stationary_intensities[t])
        add_particles(frame, rw_positions[t], rw_intensities[t])
        add_particles(frame, agg_positions[t], agg_intensities[t])

        # Crop the border off: light that fell outside the image is discarded, to handle mobile contaminants getting close to the border of the simulated video (where no other particles are simulated) smoothly by padding and then cutting the edges off.
        frame = frame[radius:-radius, radius:-radius]

        # Noise Simulation
        frame += background_level
        mean_photons = np.clip(frame * photon_scale, 0, None)
        photon_counts = rng.poisson(mean_photons).astype(np.float32)
        read_noise = rng.normal(0.0, read_noise_sigma, photon_counts.shape)
        
        frames.append(np.clip(photon_counts + read_noise, 0, None))
        
    return np.array(frames)


###PARAMETERS###

# Global
global_cfg = sim_config["global_settings"]
n_frames         = global_cfg["n_frames"]
img_size         = global_cfg["img_size"]
background_level = global_cfg["background_level"]
read_noise       = global_cfg["read_noise"]
margin           = global_cfg["margin"]
seed             = global_cfg["seed"] + int(rep) 

# Stationary
stat_cfg = sim_config["stationary"]
num_stationary = stat_cfg["count"] if stat_cfg["activate"] else 0
stat_fret_means   = stat_cfg["params"]["fret_means"]
stat_fret_stds    = stat_cfg["params"]["fret_stds"]
stat_brightness_range = stat_cfg["params"]["brightness_range"]
stat_trans_mat    = np.array(stat_cfg["params"]["trans_matrix"])
stat_pi           = stat_cfg["params"]["pi"]

# Aggregate
agg_cfg = sim_config["aggregate"]
num_aggregate = agg_cfg["count"] if agg_cfg["activate"] else 0
aggpair_count_range     = agg_cfg["params"]["aggpair_count_range"]
aggpair_brightness_mean = agg_cfg["params"]["aggpair_brightness_mean"]
aggpair_brightness_cv   = agg_cfg["params"]["aggpair_brightness_cv"]
aggpair_pi              = agg_cfg["params"]["aggpair_pi"]
aggpair_fret_means      = agg_cfg["params"]["aggpair_fret_means"]
aggpair_fret_stds       = agg_cfg["params"]["aggpair_fret_stds"]
aggpair_trans_matrix    = agg_cfg["params"]["aggpair_trans_matrix"]
jitter_std              = agg_cfg["params"]["jitter_std"]

# Moving
mov_cfg = sim_config["moving"]
num_randomwalk = mov_cfg["count"] if mov_cfg["activate"] else 0
mov_brightness_range = mov_cfg["params"]["brightness_range"]
mov_step       = mov_cfg["params"]["step_size"]
mov_life_min   = mov_cfg["params"]["min_life"]
mov_life_max   = mov_cfg["params"]["max_life"]
mov_fade_len   = mov_cfg["params"]["fade_len"]
# Settings
min_distance_stat = 7
min_distance_agg  = 4
gauss_sigma = 1.5
photon_scale = 500


###EXECUTION###


#One random number generator for the whole simulation, so the same seed always gives the same video
rng = np.random.default_rng(seed)
#This rng is used for all random generation for reproducability
print(f"Seed: {seed}")

#Generate Positions
max_min_distance = max(min_distance_stat, min_distance_agg)

stationary_positions, rw_positions, rw_starts, rw_ends, agg_positions = generate_valid_simulation_tracks(
    n_stat=num_stationary,
    n_rw=num_randomwalk,
    n_agg=num_aggregate,
    n_frames=n_frames,
    img_size=img_size,
    margin=margin,
    min_distance=max_min_distance,
    rw_step_size=mov_step,
    rw_min_life=mov_life_min,
    rw_max_life=mov_life_max,
    rng=rng
)

#Generate Intensities
stationary_intensities = generate_stationary_intensities_hmm(
    n_frames=n_frames, num_particles=num_stationary, pi=stat_pi, trans_matrix=stat_trans_mat,
    fret_means=tuple(stat_fret_means), fret_stds=tuple(stat_fret_stds),
    min_total_brightness=stat_brightness_range[0], max_total_brightness=stat_brightness_range[1], rng=rng
)

rw_intensities = generate_moving_intensities(
    n_frames, num_particles=num_randomwalk, t_starts=rw_starts, t_ends=rw_ends,
    min_brightness=mov_brightness_range[0], max_brightness=mov_brightness_range[1], fade_len=mov_fade_len, rng=rng
)

agg_intensities = generate_aggregate_intensities(
    n_frames, num_aggregate, aggpair_count_range, aggpair_brightness_mean, aggpair_brightness_cv,
    aggpair_pi, aggpair_fret_means, aggpair_fret_stds, aggpair_trans_matrix, jitter_std, rng=rng
)

#Render Frames
frames = render_frames(
    stationary_positions, rw_positions, agg_positions,
    stationary_intensities, rw_intensities, agg_intensities,
    num_stationary, num_randomwalk, num_aggregate,
    n_frames, img_size, gauss_sigma,
    photon_scale=photon_scale, read_noise_sigma=read_noise,
    background_level=background_level, rng=rng
)

#Save Video
# Move Channel dim to index 1: (T, H, W, C) -> (T, C, H, W)
frames_to_save = np.moveaxis(frames, -1, 1)
tiff.imwrite(
    path_tiff, 
    frames_to_save.astype(np.float32), 
    imagej=True, metadata={'axes': 'TCYX'}
)

#Save Ground Truth
#Stationary: Frame 0 position
if num_stationary > 0:
    stationary_starts_yx = stationary_positions[0][:, [1, 0]]
else:
    stationary_starts_yx = np.empty((0, 2))


#Random Walk: Mean position over valid track
if num_randomwalk > 0:
    rw_means_xy = np.nanmean(rw_positions, axis=0) 
    rw_starts_yx = rw_means_xy[:, [1, 0]]
else:
    rw_starts_yx = np.empty((0, 2))

# Aggregate: Frame 0 position
if num_aggregate > 0:
    agg_starts_yx = agg_positions[0][:, [1, 0]]
else:
    agg_starts_yx = np.empty((0, 2))

to_stack = []
if num_stationary > 0: to_stack.append(stationary_starts_yx)
if num_aggregate > 0: to_stack.append(agg_starts_yx)
if num_randomwalk > 0: to_stack.append(rw_starts_yx)

if to_stack:
    full_start_positions = np.vstack(to_stack)
else:
    full_start_positions = np.empty((0, 2))

np.save(path_coords, full_start_positions.astype(np.float32))

#Save Classes
stationary_classes = np.array([0] * num_stationary)
agg_classes = np.array([1] * num_aggregate)
rw_classes = np.array([2] * num_randomwalk)
full_classes = np.concatenate([stationary_classes, agg_classes, rw_classes]) 
np.save(path_classes, full_classes)

print(f"Data simulation #{rep} completed and saved.")

#Print SNR:
avg_stat_brightness = np.mean(stat_brightness_range)
#Get Signal S
#Convert total volume to peak height: Height = Volume / (2 * pi * sigma^2)
peak_height_norm = 1.0 / (2 * np.pi * gauss_sigma**2)
S_peak_photons = avg_stat_brightness * peak_height_norm * photon_scale
#Get Background B
B_photons = background_level * photon_scale
#Calculate SNR = S / sqrt(S + B + ReadNoise^2)
#Note: (S + B) is the Photon Shot Noise Variance, as shot noise follows poisson statistics, where the mean = variance.
snr = S_peak_photons / np.sqrt(S_peak_photons + B_photons + read_noise**2)
print(f"SNR SIMULATION PROFILE: {snr:.2f}")

#Donor-channel SNR: both detectors only use the donor, which gets a fraction (1 - E) of the light.
#Time-averaged donor fraction from the long-run state distribution of the two-state HMM
p01, p10 = stat_trans_mat[0, 1], stat_trans_mat[1, 0]
long_run = np.array([p10, p01]) / (p01 + p10)  # fraction of time in state 0 (low FRET) and state 1 (high FRET)
donor_fraction = np.sum(long_run * (1 - np.array(stat_fret_means)))
S_donor_photons = S_peak_photons * donor_fraction
snr_donor = S_donor_photons / np.sqrt(S_donor_photons + B_photons + read_noise**2)
#Average image over all frames (what the detectors use): noise drops by sqrt(n_frames), a static spot's signal stays
snr_donor_avg = snr_donor * np.sqrt(n_frames)
print(f"Donor-channel SNR per frame: {snr_donor:.2f} (donor fraction {donor_fraction:.2f})"
      f"donor average image ({n_frames} frames): {snr_donor_avg:.1f}")