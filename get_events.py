import pandas as pd
def get_events(dataset_name, outfile=None):
    d_id = ms.Dataset.query.filter_by(name=dataset_name).one().id
    stim_ids = ms.Stimulus.query.filter_by(
        dataset_id=d_id, mimetype='image/png').with_entities('id')
    unique_runs = ms.Run.query.filter_by(dataset_id=d_id).distinct('number').all()
    all_events = []
    for r in unique_runs:
        rss = ms.RunStimulus.query.filter_by(run_id=r.id).filter(
            ms.RunStimulus.stimulus_id.in_(stim_ids)).all()
        for rs in rss:
            all_events.append((r.number, rs.onset, rs.duration, rs.stimulus.path))
    all_events = pd.DataFrame(all_events, columns=['run_number', 'onset', 'duration', 'path'])
    if all_events is not None:
        all_events.to_csv(outfile, index=False)
    return all_events
