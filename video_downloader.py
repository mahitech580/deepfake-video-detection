import yt_dlp
import requests
import tempfile
import os
from urllib.parse import urlparse

class VideoDownloader:
    def __init__(self):
        self.ydl_opts = {
            'format': 'best[height<=720]',  # Limit to 720p for faster processing
            'outtmpl': '%(title)s.%(ext)s',
            'quiet': True,
            'no_warnings': True,
        }
    
    def is_youtube_url(self, url):
        """Check if URL is from YouTube"""
        youtube_domains = ['youtube.com', 'youtu.be', 'm.youtube.com']
        parsed = urlparse(url)
        return any(domain in parsed.netloc for domain in youtube_domains)
    
    def is_direct_video_url(self, url):
        """Check if URL is a direct video file"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
        parsed = urlparse(url)
        return any(parsed.path.lower().endswith(ext) for ext in video_extensions)
    
    def download_youtube_video(self, url):
        """Download video from YouTube using yt-dlp"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                self.ydl_opts['outtmpl'] = os.path.join(temp_dir, '%(title)s.%(ext)s')
                
                with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                    # Get video info first
                    info = ydl.extract_info(url, download=False)
                    title = info.get('title', 'video')
                    duration = info.get('duration', 0)
                    
                    # No duration limit
                    
                    # Download video
                    ydl.download([url])
                    
                    # Find downloaded file
                    for file in os.listdir(temp_dir):
                        if file.endswith(('.mp4', '.webm', '.mkv', '.avi')):
                            downloaded_path = os.path.join(temp_dir, file)
                            
                            # Copy to new temp file
                            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                            with open(downloaded_path, 'rb') as src:
                                temp_file.write(src.read())
                            temp_file.close()
                            
                            return temp_file.name, title
                    
                    raise Exception("No video file found after download")
                    
        except Exception as e:
            raise Exception(f"YouTube download failed: {str(e)}")
    
    def download_direct_video(self, url):
        """Download video from direct URL"""
        try:
            response = requests.head(url, timeout=10)
            content_length = int(response.headers.get('content-length', 0))
            
            # No file size limit
            
            # Download video
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file.close()
            
            filename = os.path.basename(urlparse(url).path) or "video"
            return temp_file.name, filename
            
        except Exception as e:
            raise Exception(f"Direct download failed: {str(e)}")
    
    def download_video(self, url):
        """Download video from URL (YouTube or direct)"""
        if self.is_youtube_url(url):
            return self.download_youtube_video(url)
        elif self.is_direct_video_url(url):
            return self.download_direct_video(url)
        else:
            # Try as YouTube first, then direct
            try:
                return self.download_youtube_video(url)
            except:
                return self.download_direct_video(url)