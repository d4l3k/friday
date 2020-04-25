from gphotos.restclient import RestClient, log
from gphotos.authorize import Authorize
from pathlib import Path
import requests

classes = {'oldpee', 'newpee', 'clear', 'poop'}

def trace(*args):
    pass
log.trace = trace

scope = [
    "https://www.googleapis.com/auth/photoslibrary.readonly",
]
photos_api_url = (
    "https://photoslibrary.googleapis.com/$discovery"
    "/rest?version=v1"
)
credentials_file = Path(".gphotos.token")
secret_file = Path("client_secret.json")
auth = Authorize(
    scope, credentials_file, secret_file, 3
)
auth.authorize()
client = RestClient(photos_api_url, auth.session)


class RespIter:
    def __init__(self, method, key, expand=True, **kwargs):
        self.method = method
        self.key = key
        self.kwargs = kwargs
        self.json = None
        self.expand = expand
        self.items = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.json is None or len(self.items) == 0:
            if self.json:
                token = self.json.get("nextPageToken")
                if not token:
                    raise StopIteration
                self.kwargs['pageToken'] = token
            if self.expand:
                resp = self.method(**self.kwargs)
            else:
                resp = self.method(self.kwargs)
            if not resp:
                raise StopIteration
            self.json = resp.json()
            self.items = self.json.get(self.key)

        if len(self.items) == 0:
            raise StopIteration

        item = self.items[0]
        self.items = self.items[1:]
        return item



print('fetching class ids...')
class_ids = {}

for album in RespIter(client.albums.list.execute, 'albums', pageSize=50):
    name = album.get('title', '').lower()
    if name in classes:
        class_ids[name] = album['id']


assert len(class_ids) == len(classes)
print(f"class_ids {class_ids}")

for name, id in class_ids.items():
    print(f"downloading {name}")
    for item in RespIter(client.mediaItems.search.execute, 'mediaItems', expand=False, albumId=id, pageSize=100):
        type = 'dv' if '.mp4' in item['filename'] else 'd'
        url = item['baseUrl'] + '=' + type
        filename = f"data/{name}/{item['filename']}"
        print(f"downloading {filename}")
        r = requests.get(url, allow_redirects=True)
        r.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(r.content)
