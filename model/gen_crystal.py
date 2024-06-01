import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from ase.io import read,write
import matplotlib.pyplot as plt
from pyxtal.symmetry import Group
from ase.geometry import get_distances
from ase.neighborlist import neighbor_list
from sklearn.cluster import KMeans
from itertools import permutations
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
np.set_printoptions(suppress=True, precision=4)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

import warnings 
warnings.filterwarnings("ignore")
Density_max = 0.2
Density_min = 0.015
import configparser
config = configparser.ConfigParser()
config.read('gen.ini')
class Data_Generator:
    def __init__(self, data_path, generator_model_path, z_dim, output_dim):
        self.data_path = data_path
        self.generator_model_path = generator_model_path
        self.z_dim = z_dim
        self.output_dim = output_dim
        self.scaler = StandardScaler()
        self.generator = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self):
        return pd.read_csv(self.data_path)

    def load_generator_model(self):
        self.generator = Generator(input_dim=self.z_dim, output_dim=self.output_dim)
        self.generator.load_state_dict(torch.load(self.generator_model_path))
        self.generator.to(self.device)
        self.generator.eval()

    def generate_samples(self, num_samples):
        z = Variable(torch.randn(num_samples, self.z_dim)).to(self.device)
        with torch.no_grad():
            generated_samples = self.generator(z).cpu()  # Move the generated samples to CPU if necessary
        return generated_samples

    def process_data(self, datas, generated_samples):
        datas = self.scaler.fit_transform(datas)
        real_data = self.scaler.inverse_transform(generated_samples)
        return np.round(real_data, 2)

    def print_first_sample(self, real_data):
        print(" ".join(list(map(str, real_data[0]))))

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv1d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv1d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        proj_query = self.query_conv(x).view(x.shape[0], -1, x.shape[2])
        proj_key = self.key_conv(x).view(x.shape[0], -1, x.shape[2])
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(x.shape[0], -1, x.shape[2])

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(x.shape[0], x.shape[1], x.shape[2])
        out = self.gamma * out + x
        return out, attention
# Define the Generator network
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.network(z)

# Define the Critic network
# Critic network with Self-Attention
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.self_attention = SelfAttention(in_dim=128)
        self.final_layer = nn.Linear(128, 1)

    def forward(self, x):
        x = self.network(x)
        x, attention = self.self_attention(x.unsqueeze(2))  # Add dummy dimension for self-attention
        x = x.squeeze(2)  # Remove dummy dimension after self-attention
        x = self.final_layer(x)
        return x
class Stru_Generator:
    def __init__(self, gpgh, elemset, synthetic_data,N_max_per_cell):
        self.gpgh = gpgh
        self.elemset = elemset
        self.synthetic_data = synthetic_data
        self.N_max_per_cell = N_max_per_cell
        self.gen_files = f"gen_files_{self.gpgh}"
        self._create_directory()
        self.filtered_permutations = self._filter_permutations()

    def _create_directory(self):
        if not os.path.exists(self.gen_files):
            os.mkdir(self.gen_files)

    def _filter_permutations(self):
        unique_permutations = set(permutations(self.elemset))
        return {perm for perm in unique_permutations if perm[-1] == 'O'}

    def process_synthetic_data(self):
        
        # for elem_permutation in list(self.filtered_permutations):
        #     elemlist = list(elem_permutation)
        for index in range(len(self.synthetic_data)):
            sd_data = np.array(self.synthetic_data.iloc[index, :])
            sd_data = np.round(sd_data, 3)
            type_counter = 0
            for elem_permutation in list(self.filtered_permutations):
                elemlist = list(elem_permutation)
                # print(index,elemlist)
                try:
                    # print(sd_data,"|", str(index)+"t"+str(type_counter), "|",elemlist, "|",self.gen_files, "|",self.gpgh)
                    _ = self.get_sample_index(sd_data, str(index)+"t"+str(type_counter), elemlist, self.gen_files, self.gpgh)
                    # if _:
                    #     pass
                    # else:
                    #     continue
                except Exception as e:
                    print(f"Error processing data: {e}")
                type_counter += 1
    def get_sample_index(self,nsd_data,sample_index,elemlist,gen_files,spgh=None):
        sample_index = sample_index
        sd_data      = nsd_data
        cif_name     = f"{gen_files}/{sample_index}_{spgh}.cif"
        if spgh is None:
            for spgh in tqdm(range(1,231)):
                #spgh =  2
                oper_set = get_symmetry_oper(spgh,None)
                centers = how_much_center(spgh)
                cif_name = f"{gen_files}/{sample_index}_{spgh}.cif"
                if centers >1:
                    # print(f"spgh{spgh} with {centers} centers")
                    site_a = sd_data[7:10]
                    site_b = sd_data[10:13]
                    site_c = sd_data[13:16]
                    site_d = sd_data[16:19]
                    aabbcc = sd_data[:7]
                    site_a_fix = kmcluster(site_a, oper_set,centers)
                    site_b_fix = kmcluster(site_b, oper_set,centers)
                    site_c_fix = kmcluster(site_c, oper_set,centers)
                    site_d_fix = kmcluster(site_d, oper_set,centers)
                    # print("before",site_a, site_b, site_c)
                    # print("fixed",site_a_fix, site_b_fix, site_c_fix)
                    #print("*"*99)
                    #print("brfore",aabbcc)
                    _,aabbcc = spg2latt_refine(spgh, aabbcc[0], aabbcc[1], aabbcc[2], aabbcc[3], aabbcc[4], aabbcc[5])
                    #print("fixed",aabbcc)
                    #print("centers >1")
                    cif_writer(spgh, aabbcc, oper_set, cif_name,site_a_fix, site_b_fix, site_c_fix,site_d_fix,elemlist)
                else:
                    site_a = process_array(sd_data[7:10])
                    site_b = process_array(sd_data[10:13])
                    site_c = process_array(sd_data[13:16])
                    site_d = process_array(sd_data[16:19])
                    aabbcc = sd_data[:7]
                    _,aabbcc = spg2latt_refine(spgh, aabbcc[0], aabbcc[1], aabbcc[2], aabbcc[3], aabbcc[4], aabbcc[5])
                    # print(site_d)
                    cif_writer(spgh, aabbcc, oper_set, cif_name,site_a, site_b, site_c,site_d,elemlist)
                read_atom = read(cif_name)
                N_atoms = len(read_atom)
                if N_atoms > self.N_max_per_cell:
                    continue
                V_atoms = read_atom.get_volume()
                density = N_atoms/V_atoms
                if density < Density_min or density > Density_max:
                    try:
                        os.remove(cif_name)
                    except:
                        pass
                if not neighbor_ok(read_atom):
                    try:
                        os.remove(cif_name)
                        #print("pass")
                    except:
                        pass
                continue
        else:
            spgh = spgh
            oper_set = get_symmetry_oper(spgh,None)
            centers = how_much_center(spgh)
            # centers = 8  # test
            #print(centers)
            if centers > 1:
                    site_a = sd_data[7:10]
                    site_b = sd_data[10:13]
                    site_c = sd_data[13:16]
                    site_d = sd_data[16:19]
                    site_a_fix = kmcluster(site_a, oper_set,centers)
                    site_b_fix = kmcluster(site_b, oper_set,centers)
                    site_c_fix = kmcluster(site_c, oper_set,centers)
                    site_d_fix = kmcluster(site_d, oper_set,centers)
                    aabbcc = sd_data[:7]
                    _,aabbcc = spg2latt_refine(spgh, aabbcc[0], aabbcc[1], aabbcc[2], aabbcc[3], aabbcc[4], aabbcc[5])
                    cif_writer(spgh, aabbcc, oper_set, cif_name,site_a_fix, site_b_fix, site_c_fix,site_d_fix,elemlist)
            if centers == 1:
                    # print(sd_data[7:10])
                    site_a = process_array(sd_data[7:10])
                    site_b = process_array(sd_data[10:13])  
                    site_c = process_array(sd_data[13:16])
                    site_d = process_array(sd_data[16:19])
                    # print(site_a, site_b, site_c,site_d)
                    aabbcc = sd_data[:7]
                    _,aabbcc = spg2latt_refine(spgh, aabbcc[0], aabbcc[1], aabbcc[2], aabbcc[3], aabbcc[4], aabbcc[5])
                    cif_writer(spgh, aabbcc, oper_set, cif_name,site_a, site_b, site_c,site_d,elemlist)
            read_atom = read(cif_name)
            N_atoms = len(read_atom)
            if N_atoms > self.N_max_per_cell:
                # continue statement removed
                os.remove(cif_name)
                #return False
            else:
                V_atoms = read_atom.get_volume()
                density = N_atoms/V_atoms
                if density < Density_min or density > Density_max:
                    try:
                        os.remove(cif_name)
                        #return False
                    except:
                        pass
                        #return False
                if not neighbor_ok(read_atom):
                    try:
                        os.remove(cif_name)
                        #return False
                    except:
                        #return True
                        pass
def neighbor_ok(crystal, Rmin=0.9, Rmax=5.0):
    indices = neighbor_list('ijd', crystal, Rmax)
    if indices[2].min() > Rmin:
        if indices[2].max() < Rmax:
            return True
        else:
            return False
    return False
def spgnb2spgh(num):
    group = Group(num)
    return group.symbol
def get_symmetry_oper(spgnumber,path):
    global wyckoff_path
    path = wyckoff_path
    oper = pd.read_csv(path)
    oper_spg = eval(oper.iloc[spgnumber,:][1])[0]
    # oper_spg = [j for i in tmp for j in i]
    # print(oper_spg)
    # oper_spg = [i for i in oper_spg]
    return oper_spg
def how_much_center(spgh):
    in_put = spgh
    oper_set = get_symmetry_oper(spgh,wyckoff_path)
    x,y,z = 0,0,0
    center_list = []
    for _ in range(len(oper_set)):
        data = eval(oper_set[_])

        if data[0] < 0:
            data[0] = data[0]%1
        if data[1] < 0:
            data[1] = data[1]%1
        if data[2] < 0:
            data[2] = data[2]%1
        center_list.append(data)
        #print(len(set(center_list)))
    return len(set(center_list))
def cif_writer(spgh, aabbcc, oper_set, cif_name,site_a, site_b, site_c,site_d,elemlist=['Ba','Ru','O','O']):
    write_oper_set = ''
    for i, item in enumerate(oper_set, start=1):
        write_oper_set += f"{i} \"{item}\" \n"
    stru_cif = f"""# generated using CGWWGAN
    data_generated_sutianhao@shu.edu.cn
    _symmetry_space_group_name_H-M {spgnb2spgh(spgh)}
    _cell_length_a   {aabbcc[0]}
    _cell_length_b   {aabbcc[1]}
    _cell_length_c   {aabbcc[2]}
    _cell_angle_alpha   {aabbcc[3]}
    _cell_angle_beta   {aabbcc[4]}
    _cell_angle_gamma   {aabbcc[5]}
    loop_
    _symmetry_equiv_pos_site_id
    _symmetry_equiv_pos_as_xyz
    {write_oper_set}
    loop_
    _atom_site_type_symbol
    _atom_site_label
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_occupancy
    {elemlist[0]}  {elemlist[0]}  {site_a[0]}  {site_a[1]}  {site_a[2]}  1
    {elemlist[1]}  {elemlist[1]}  {site_b[0]}  {site_b[1]}  {site_b[2]}  1
    {elemlist[2]}  {elemlist[2]}  {site_c[0]}  {site_c[1]}  {site_c[2]}  1
    {elemlist[3]}  {elemlist[3]}  {site_d[0]}  {site_d[1]}  {site_d[2]}  1"""
    #print(f"Generated CIF file: {cif_name}")
    with open(f'{cif_name}', "w") as f:
        f.write(stru_cif)
    if float(site_d[0]) < -1 or float(site_d[1]) < -1 or float(site_d[2]) < -1 or float(site_d[0]) > 1 or float(site_d[1]) > 1 or float(site_d[2]) > 1:
        with open(f'{cif_name}', "r+") as f:
            lines = f.readlines()
            f.seek(0)
            f.truncate()
            for line in lines[:-1]:
                f.write(line)
    return cif_name
def process_array(arr):
    processed_arr = []
    for element in arr:
        # 
        if element > 1 or element < 0:
            element = element % 1
        # 
        if abs(element - 0) < 0.1:
            processed_arr.append(0)
        elif abs(element - 0.25) < 0.1:
            processed_arr.append(0.25)
        elif abs(element - 0.5) < 0.1:
            processed_arr.append(0.5)
        else:
            processed_arr.append(element)
    return np.array(processed_arr)
def kmcluster(site_a, oper_set,n_clusters):
    in_put = site_a
    x,y,z = site_a
    expand_list = []
    for i in range(len(oper_set)):
        data = eval(oper_set[i])
        expand_list.append(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(expand_list)
    out_put = kmeans.cluster_centers_[0]
    #print(in_put,out_put)
    return out_put
def cnvtlatticeparamter2matrix(a,b,c,alpha,beta,gamma):
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)
    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    val = abs(val) if abs(val) < 1. else  1.
    gammas   = np.arccos(val)
    cellmat  =[[ a * sin_beta,                  0.0,                           a * cos_beta],
               [-b * sin_alpha * np.cos(gammas),b * sin_alpha * np.sin(gammas),b * cos_alpha],
               [0.0,                            0.0,                           c]]
    return np.array(cellmat,dtype=float)

def spg2latt_type(spgnum):
    return df_latt.iloc[spgnum-1,0]
def spg2latt_refine(spgnum, a, b, c, alpha, beta, gamma):
    a = float(a)
    b = float(b)
    c = float(c)
    lat_types = {
        'cubic': (a, a, a, 90, 90, 90),
        'triclinic': (a, b, c, alpha, beta, gamma),
        'monoclinic': (a, b, c, 90, beta, 90),
        'orthorhombic': (a, b, c, 90, 90, 90),
        'tetragonal': (a, a, c, 90, 90, 90),
        'hexagonal': (a, a, c, 90, 90, 120), 
        'trigonal': (a, a, a, alpha, beta, gamma)
    }
    #print(spg2latt_type(spgnum))
    cell = cnvtlatticeparamter2matrix(*lat_types[spg2latt_type(spgnum)])
    return cell,lat_types[spg2latt_type(spgnum)]

wyckoff_path = ""
global df_latt
data_generator_config = config['data_generator']
data_generator = Data_Generator(
    data_path=data_generator_config['data_path']+"mpdata.csv",
    generator_model_path=data_generator_config['generator_model_path'],
    z_dim=int(data_generator_config['z_dim']),
    output_dim=int(data_generator_config['output_dim'])
)
df_latt = pd.read_csv(data_generator_config['data_path']+"lattice_type.csv")
wyckoff_path = data_generator_config['data_path']+"wyckoff_list.csv"

general_config = config['general']
num_samples_tryto_generate = int(general_config['num_samples_tryto_generate'])
stru_generator_config = config['stru_generator']
gpgh = int(stru_generator_config['spgn'])
elemset = stru_generator_config['elemset'].split(', ')
N_max_per_cell = int(stru_generator_config['N_max_per_cell'])
datas = data_generator.load_data()
data_generator.load_generator_model()
generated_samples = data_generator.generate_samples(num_samples_tryto_generate)
real_data = data_generator.process_data(datas, generated_samples)
# data_generator.print_first_sample(real_data)
synthetic_data = pd.DataFrame(real_data)
synthetic_data=synthetic_data
data_generator = Stru_Generator(gpgh=gpgh, elemset=elemset, synthetic_data=synthetic_data,N_max_per_cell=N_max_per_cell)
data_generator.process_synthetic_data()