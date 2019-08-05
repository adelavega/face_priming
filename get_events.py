import pandas as pd
def get_events(dataset_name, outfile=None):
    d_id = ms.Dataset.query.filter_by(name=dataset_name).one().id
    stim_ids = ms.Stimulus.query.filter_by(
        dataset_id=d_id, mimetype='image/png').with_entities('id')
    unique_videos = ms.Stimulus.query.filter_by(dataset_id=d_id, mimetype='video/mp4').all()

    all_events = []
    for v in unique_videos:
        r_id = ms.RunStimulus.query.filter_by(stimulus_id=v.id).first().run_id
        r = ms.Run.query.filter_by(id=r_id).one()
        rss = ms.RunStimulus.query.filter_by(run_id=r.id).filter(
            ms.RunStimulus.stimulus_id.in_(stim_ids)).all()
        original_stim = v.path.split('/')[-1]
        for rs in rss:
            all_events.append((r.number, original_stim, rs.onset, rs.duration, rs.stimulus.path))
    all_events = pd.DataFrame(all_events, columns=['run_number', 'original_stim', 'onset', 'duration', 'path'])
    all_events = all_events.sort_values(['original_stim', 'onset'])

    if all_events is not None:
        all_events.to_csv(outfile, index=False)
        
    return all_events
