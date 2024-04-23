from utils import (
    get_mlbam_key,
    process_statcast_data,
)
from pybaseball import statcast_pitcher

pitcher_ids = [
    get_mlbam_key("Imanaga", "Shota"),
    get_mlbam_key("Hendricks", "Kyle"),
    get_mlbam_key("Steele", "Justin"),
    get_mlbam_key("Wicks", "Jordan"),
    get_mlbam_key("Assad", "Javier"),
    get_mlbam_key("Brown", "Ben"),
    get_mlbam_key("Crochet", "Garrett"),
    get_mlbam_key("Soroka", "Michael"),
    get_mlbam_key("Fedde", "Erick"),
    get_mlbam_key("Flexen", "Chris"),
    get_mlbam_key("Kopech", "Michael"),
]

for pitcher_id in pitcher_ids:
    df = statcast_pitcher("2022-01-01", "2024-04-17", pitcher_id)
    df = process_statcast_data(df)
    df.to_parquet(f"data/{pitcher_id}.parquet")