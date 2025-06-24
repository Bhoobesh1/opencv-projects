from bing_image_downloader import downloader

downloader.download("Cristiano Ronaldo", limit=3, output_dir='faces', adult_filter_off=True, force_replace=False, timeout=40)
downloader.download("Dhoni",limit=3,output_dir='faces',adult_filter_off=True, force_replace=False,timeout=40)