from pytube import YouTube
from pathlib import Path
import argparse
parser = argparse.ArgumentParser()


parser.add_argument('--name', required = True, 
					help = 'output_video_name')
parser.add_argument('--url', required = True, 
					help = 'video_url')

opt = parser.parse_args()
video_name=opt.name+".mp4"
url=opt.url
yt = YouTube(url)
yt.streams.first().download("data/", 'download_mv')
