import requests
from tqdm import tqdm
raise NotImplementedError("NRAO hasn't delivered this yet, ...")

S = requests.Session()

S.cookies['mod_auth_openidc_session'] = '7934a2d8-96c2-468e-bc9c-23648adebcdf'

# uid://A001/X3833/X67e2
filenames = [
'2025.1.00021.S_uid___A001_X3833_X67e2',
]
'uid://A001/X3833/X67e4'

root = 'https://bulk.cv.nrao.edu/almadata/proprietary/2025.1.0021.S/X67e2/'
# https://almascience.nrao.edu/aq?member_ous_id=uid://A001/X3845/Xa31

for filename in filenames:
    with open(filename, 'wb') as fh:
        response = S.get(f'{root}/{filename}', stream=True)

        for chunk in tqdm(response.iter_content(chunk_size=8192)):
            if chunk:
                fh.write(chunk)
