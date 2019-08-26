import pandas as pd
import numpy as np

def _face_detection(paths):
    face_detection = pd.DataFrame(paths, columns=['path'])
    face_detection['path'] = face_detection.apply(lambda x: '/file-data/stimuli/' + x['path'].split('/')[-1], axis=1)

    face_detection['face_count'] = 1
    face_detection = face_detection.groupby('path').sum().reset_index()
    face_detection['any_faces'] = 1    
    return face_detection

def _get_wide_cluster_ids(clusters):
    # 2. IDs by time
    frame_char = []
    for i, c in enumerate(clusters):
        cn = i + 1
        f_paths = list(zip(*c))[0]
        frame_char += list(zip(['id_' + str(cn)] * len(f_paths), f_paths ))

    cluster_ids = pd.DataFrame(frame_char, columns=['face_cluster_id', 'path'])

    cluster_ids['path'] = cluster_ids.apply(lambda x: '/file-data/stimuli/' + x['path'].split('/')[-1], axis=1)
    cluster_ids['value'] = 1

    # Make wide
    cluster_ids_wide = cluster_ids.reset_index().pivot_table(index=['path', 'index'], columns='face_cluster_id', values='value')

    cluster_ids_wide = cluster_ids_wide.groupby('path').max().reset_index()
    
    return cluster_ids_wide

def _first_time(row, all_faces):
    face_cols = row.loc[all_faces]
    faces = face_cols[pd.isnull(face_cols) == False]
    if faces.sum() > 0:
        for v in faces.index:
            all_faces.remove(v)
        return 1
    return np.nan

def _calc_running_time(row, last_offset):
    if row.run_number == 1:
        return row.onset
    else:
        return row.onset + last_offset[row.run_number - 1]

def _get_face_times(row, all_faces, output):
    """ For every row, then every colum, time delta since face was last shown (in cummulative time),
    and total duration face has been on screen. Then average these values across all faces shown on any frame """
    active = row.loc[all_faces]
    active = active[~active.isnull()].index

    time_since = []
    face_time_cum = []
    
    for face_ix in active:
        previous_onsets = output[(pd.isnull(output[face_ix]) == False) & (output.cum_onset < row.cum_onset)]
        if len(previous_onsets) > 0:
            time_since.append(row.cum_onset - previous_onsets.iloc[-1].cum_onset)
            face_time_cum.append(previous_onsets.duration.sum() + row.duration)
    if len(time_since) == 0:
        return pd.Series([np.nan, np.nan, np.nan, np.nan])
    else:
        return pd.Series([np.mean(time_since), np.max(time_since), np.mean(face_time_cum), np.max(face_time_cum)])

def _compute_time_dependent(subset, all_faces):
    subset['first_time_face'] = subset.apply(_first_time, axis=1, all_faces=all_faces.copy())
   
    # Each run's offset
    last_offset = subset.groupby('run_number').max()['onset'].cumsum()
    subset['cum_onset'] = subset.apply(_calc_running_time, axis=1, last_offset=last_offset)

    subset[['mean_time_since', 'max_time_since', 'mean_face_time_cum', 'max_face_time_cum']] = subset.apply(
        _get_face_times, axis=1, all_faces=all_faces.copy(), output=subset)

    return subset

def prepare_regressors(frames, paths, clusters, bad_clusters=None):
    # Do face detection
    face_detection = _face_detection(paths)
    output = pd.merge(frames, face_detection, how='outer')
    
    cluster_ids_wide = _get_wide_cluster_ids(clusters)
    # Merge face IDs into main file
    output = pd.merge(output, cluster_ids_wide, how='outer')
    output = output.sort_values(['run_number', 'onset']).reset_index().drop(columns='index')
    
    all_faces = ["id_" + str(i) for i in range(1, len(clusters) + 1)]
    for bad in bad_clusters:
        all_faces.remove(f"id_{bad}")
        
    output = output.groupby('cond').apply(_compute_time_dependent, all_faces)

    output['log_mean_time_since'] = np.log(output.mean_time_since)
    output['log_mean_face_time_cum'] = np.log(output.mean_face_time_cum)
    output['log_max_time_since'] = np.log(output.max_time_since)
    output['log_max_face_time_cum'] = np.log(output.max_face_time_cum)

    
    return output