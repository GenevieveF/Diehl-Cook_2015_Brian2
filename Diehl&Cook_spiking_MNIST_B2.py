'''
Created on 15.12.2014

@author: Peter U. Diehl

Updated by Genevieve Fahey to use Python 3.x and Brian 2 instead of Python 2.x and Brian 1
'''

import matplotlib.cm as cmap
import os.path
import pickle
import brian2 as b
from struct import unpack
from brian2 import *
from brian2tools import *
from datetime import datetime

# specify the location of the MNIST data
MNIST_data_path = 'MNIST/'

#------------------------------------------------------------------------------
# functions
#------------------------------------------------------------------------------
def get_labeled_data(picklename, bTrain = True):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename,'rb'))
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(MNIST_data_path + 'train-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 'train-labels.idx1-ubyte','rb')
        else:
            images = open(MNIST_data_path + 't10k-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 't10k-labels.idx1-ubyte','rb')
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0]

        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
        for i in range(N):
            if i % 1000 == 0:
                print("i: %i" % i)
            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in range(cols)]  for unused_row in range(rows) ]
            y[i] = unpack('>B', labels.read(1))[0]

        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open('%s.pickle' % picklename, 'wb'))
    return data

def get_matrix_from_file(fileName):
    offset = len(ending) + 4
    if fileName[-4-offset] == 'X':
        n_syn = n_input * n_e
    else:
        if fileName[-3-offset]=='e':
            n_syn = n_e
        else:
            n_syn = n_i * n_e
    readout = np.load(fileName)
    print(readout.shape, fileName)
    value_arr = np.zeros(n_syn)
    if readout.shape == (n_input * n_e,):
        value_arr = readout
    else:
        if not readout.shape == (0,):
            value_arr = readout[:,2]
    return value_arr


def save_connections(ending = ''):
    print('save connections')
    connMatrix = np.copy(conn_ee.w)
    np.save(data_path + 'weights/' + 'XeAe' + save_name + ending, connMatrix)

def save_theta(ending = ''):
    print('save theta')
    np.save(data_path + 'weights/theta_A' + save_name + ending, group_ne.theta)

def normalize_weights():
    temp_conn = np.copy(conn_ee.w)
    temp_conn.shape = (n_input, n_e) # new
    colSums = np.sum(temp_conn, axis = 0)
    colFactors = weight_ee/colSums
    np.save(data_path + 'activity/neuron71', temp_conn[:,71])
    for j in range(n_e):
        temp_conn[:,j] *= colFactors[j]
    temp_conn.shape = (n_input * n_e)
    conn_ee.w = temp_conn[:]

def get_2d_input_weights():
    name = 'XeAe'
    wt_matrix = np.zeros((n_input, n_e))
    n_e_sqrt = int(np.sqrt(n_e))
    n_in_sqrt = int(np.sqrt(n_input))
    num_values_col = n_e_sqrt*n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))
    connMatrix = conn_ee.w
    wt_matrix = np.copy(connMatrix)

    for i in range(n_e_sqrt):
        for j in range(n_e_sqrt):
                rearranged_weights[i*n_in_sqrt : (i+1)*n_in_sqrt, j*n_in_sqrt : (j+1)*n_in_sqrt] = \
                    wt_matrix[:, i + j*n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
    return rearranged_weights


def plot_2d_input_weights(): # plot input weights does not currently work
    name = 'XeAe'
    weights = get_2d_input_weights()
    fig = b.figure(fig_num, figsize = (18, 18))
    im2 = b.imshow(weights, interpolation = "nearest", vmin = 0, vmax = wmax_ee, cmap = cmap.get_cmap('hot_r'))
    b.colorbar(im2)
    b.title('weights of connection' + name)
    fig.canvas.draw()
    return im2, fig

def update_2d_input_weights(im, fig):
    weights = get_2d_input_weights()
    im.set_array(weights)
    fig.canvas.draw()
    return im

def get_current_performance(performance, current_example_num):
    current_evaluation = int(current_example_num/update_interval)
    start_num = current_example_num - update_interval
    end_num = current_example_num
    difference = outputNumbers[start_num:end_num, 0] - input_numbers[start_num:end_num]
    correct = len(np.where(difference == 0)[0])
    performance[current_evaluation] = correct / float(update_interval) * 100
    return performance

def plot_performance(fig_num): # performance plotting not supported in this version
    num_evaluations = int(num_examples/update_interval)
    time_steps = range(0, num_evaluations)
    performance = np.zeros(num_evaluations)
    fig = b.figure(fig_num, figsize = (5, 5))
    fig_num += 1
    ax = fig.add_subplot(111)
    im2, = ax.plot(time_steps, performance) #my_cmap
    b.ylim(ymax = 100)
    b.title('Classification performance')
    fig.canvas.draw()
    return im2, performance, fig_num, fig

def update_performance_plot(im, performance, current_example_num, fig): # performance plotting not supported in this version
    performance = get_current_performance(performance, current_example_num)
    im.set_ydata(performance)
    fig.canvas.draw()
    return im, performance

def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]

def get_new_assignments(result_monitor, input_numbers):
    assignments = np.zeros(n_e)
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e
    for j in range(10):
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_assignments
        for i in range(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j
    return assignments

#------------------------------------------------------------------------------
# load MNIST
#------------------------------------------------------------------------------
start = time.time()
training = get_labeled_data(MNIST_data_path + 'training')
end = time.time()
print('time needed to load training set:', end - start)

start = time.time()
testing = get_labeled_data(MNIST_data_path + 'testing', bTrain = False)
end = time.time()
print('time needed to load test set:', end - start)


#------------------------------------------------------------------------------
# set parameters and equations
#------------------------------------------------------------------------------
test_mode = True

b.prefs.codegen.target = 'cython'  # updated from useweave, can use 'auto' as default
b.prefs.codegen.cpp.extra_compile_args_gcc = ['-ffast-math -march=native']  # updated from ggc_options

np.random.seed(0)
save_name = '_savename'  # change this for saving thetas and weights each training run
load_name = '_savename' # leave as '' to load pre-generated random weights from original repository
    # or replace with save_name for loading weights for testing
data_path = './'

timer_start = datetime.now()
print('Start time: ', timer_start)

if test_mode:
    weight_path = data_path + 'weights/'
    num_examples = 10000 * 1
    use_testing_set = True
    do_plot_performance = False
    record_spikes = True
    ee_STDP_on = False
    update_interval = num_examples
else:
    weight_path = data_path + 'random/'
    num_examples = 60000 * 3
    use_testing_set = False
    do_plot_performance = False # performance plotting not supported in this version
    if num_examples <= 60000:
        record_spikes = True
    else:
        record_spikes = True
    ee_STDP_on = True


ending = ''
n_input = 784
n_e = 400
n_i = n_e
single_example_time = 0.35 * b.second
resting_time = 0.15 * b.second
runtime = num_examples * (single_example_time + resting_time)
if num_examples <= 10000:
    update_interval = num_examples
    weight_update_interval = 20
else:
    update_interval = 10000
    weight_update_interval = 100
if num_examples <= 60000:
    save_connections_interval = 10000
else:
    save_connections_interval = 10000
    update_interval = 10000

v_rest_e = -65. * b.mV
v_rest_i = -60. * b.mV
v_reset_e = -65. * b.mV
v_reset_i = -45. * b.mV
v_thresh_e2 = -52. * b.mV
v_thresh_i = -40. * b.mV
refrac_e = 5. * b.ms
refrac_i = 2. * b.ms

weight_ee = 78.
input_intensity = 2.
delay_ee = 10. * b.ms
delay_ie = 5. * b.ms
rand_weight = 1
start_input_intensity = input_intensity

tc_pre_ee = 20*b.ms
tc_post_1_ee = 20*b.ms
tc_post_2_ee = 40*b.ms
nu_ee_pre = 0.0001      # learning rate
nu_ee_post = 0.01       # learning rate
wmax_ee = 1.0
exp_ee_pre = 0.2
exp_ee_post = exp_ee_pre
STDP_offset = 0.4

weight_ei = 10.4    # REVISE
weight_ie = 17.0
offset = 20.0*b.mV

if test_mode:
    scr_e = 'v = v_reset_e; timer = 0*ms'
else:
    tc_theta = 1e7 * b.ms
    theta_plus_e = 0.05 * b.mV
    scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'

v_thresh_e = 'v > (theta - offset + v_thresh_e2)'


neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-100.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''
if test_mode:
    neuron_eqs_e += '\n  theta      :volt'
else:
    neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'
neuron_eqs_e += '\n  dtimer/dt = 0.1 : second'

neuron_eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-85.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''

eqs_stdp_ee = '''
                w                                      : 1
                post2before                            : 1
                dpre1/dt   =   -pre1/(tc_pre_ee)       : 1 (clock-driven)
                dpost1/dt  = -post1/(tc_post_1_ee)     : 1 (clock-driven)
                dpost2/dt  = -post2/(tc_post_2_ee)     : 1 (clock-driven)
            '''             # added w
eqs_stdp_pre_ee = '''pre1 = 1
                  w -= nu_ee_pre * post1
                  w = clip(w, 0, wmax_ee)
                  ge_post += w       
                  '''
            # added ge_post for updating exc. neurons, and clip for wmax
eqs_stdp_post_ee = '''
                post2before = post2
                w += nu_ee_post * pre1 * post2before
                w = clip(w, 0, wmax_ee)
                post1 = 1
                post2 = 1
                '''

b.ion()
fig_num = 1
result_monitor = np.zeros((update_interval,n_e))

group_ne = b.NeuronGroup(n_e, neuron_eqs_e, method = 'euler', threshold= v_thresh_e, refractory= refrac_e, reset= scr_e)
group_ni = b.NeuronGroup(n_i, neuron_eqs_i, method = 'euler', threshold= 'v > v_thresh_i', refractory= refrac_i, reset= 'v = v_reset_i')


#------------------------------------------------------------------------------
# create network population and recurrent connections
#------------------------------------------------------------------------------
print('create neuron groups')
group_ne.v = v_rest_e - 40. * b.mV
group_ni.v = v_rest_i - 40. * b.mV

if test_mode: # or weight_path[-8:] == 'weights/':
    group_ne.theta = np.load(weight_path + 'theta_A' + load_name + ending + '.npy') * b.volt
else:
    group_ne.theta = np.ones(n_e) * 20.0*b.mV


print('create recurrent connections')
# update note: uses Brian 2 one-to-one and i!=j connections with the same weights rather
# than loading in weight files with all-to-all equivalent weight files as in the original code
conn_ei = b.Synapses(group_ne, group_ni, 'w : 1', on_pre='ge_post += w')
conn_ei.connect(condition='i == j')
conn_ei.w = np.ones(len(conn_ei.w)) * weight_ei

conn_ie = b.Synapses(group_ni, group_ne, 'w : 1', on_pre='gi_post += w')
conn_ie.connect(condition='i != j')
conn_ie.delay = 'rand()*delay_ie'
conn_ie.w = np.ones(len(conn_ie.w)) * weight_ie

print('create monitors')
# rate_ei = b.PopulationRateMonitor(group_ne) # plotting for rate monitors at end of file does not work
# rate_ie = b.PopulationRateMonitor(group_ni)

if record_spikes:
    spikes_ei = b.SpikeMonitor(group_ne, record=True)
    spikes_ie = b.SpikeMonitor(group_ni, record=True)

#if record_spikes: # plotting for record_spikes not supported in this version
    # b.figure(fig_num)
    # fig_num += 1
    # b.ion()
    # b.subplot(211)
    # plot(spikes_ei.t / b.ms, spikes_ei.i, '.k')
    # xlabel('Time (ms)')
    # ylabel('Neuron index')
    # b.subplot(212)
    # plot(spikes_ie.t / b.ms, spikes_ie.i, '.k')
    # xlabel('Time (ms)')
    # ylabel('Neuron index')


#------------------------------------------------------------------------------
# create input population and connections from input populations
#------------------------------------------------------------------------------
group_in = b.PoissonGroup(n_input, 0*b.Hz)
spikes_in = b.SpikeMonitor(group_in, record=True)

print('create input connections')
if ee_STDP_on: # defaults to method = 'exact', specified method = 'euler'
    conn_ee = b.Synapses(group_in, group_ne, eqs_stdp_ee, method='euler', on_pre = eqs_stdp_pre_ee, on_post = eqs_stdp_post_ee)
    conn_ee.connect()
    #conn_ee.w = 'rand()*rand_weight' # use for random weights
    weightMatrix = get_matrix_from_file('weights/XeAe' + load_name + ending + '.npy') # loads pre-generated random initial weights
    conn_ee.w = weightMatrix
else:
    weightMatrix = get_matrix_from_file(weight_path + 'XeAe' + load_name + ending + '.npy')
    conn_ee = b.Synapses(group_in, group_ne, 'w : 1', on_pre = 'ge_post += w')
    conn_ee.connect()
    conn_ee.w = weightMatrix
    conn_ee.delay = 'rand()*delay_ee'

#------------------------------------------------------------------------------
# run the simulation and set inputs
#------------------------------------------------------------------------------
print("starting simulation")
assignments = np.zeros(n_e)
previous_spike_count = 0
previous_spikes = np.zeros(n_e)
input_numbers = [0] * num_examples
outputNumbers = np.zeros((num_examples, 10))

# if not test_mode: # input weight plotting not supported in this version
#     input_weight_monitor, fig_weights = plot_2d_input_weights()
#     fig_num += 1

#if do_plot_performance: # performance plotting not supported in this version
#    performance_monitor, performance, fig_num, fig_performance = plot_performance(fig_num)

group_in.rates = 0 * b.Hz
b.run(0 * b.second)
j = 0

while j < (int(num_examples)):
    if test_mode:
        if use_testing_set:
            rates2 = testing['x'][j%10000,:,:].reshape(n_input) / 8. * input_intensity
        else:
            rates2 = training['x'][j%60000,:,:].reshape(n_input) / 8. * input_intensity
    else:
        normalize_weights()
        rates2 = training['x'][j%60000,:,:].reshape(n_input) / 8. * input_intensity
    group_in.rates = rates2 * b.Hz
#     print('run number:', j+1, 'of', int(num_examples))
    #b.run(single_example_time, report='text')
    b.run(single_example_time)

    #if j % update_interval == 0 and j > 0: # not necessary for this script
        #assignments = get_new_assignments(result_monitor[:], input_numbers[j-update_interval : j])
    # if j % weight_update_interval == 0 and not test_mode: # input weight plotting not supported in this version
    #     update_2d_input_weights(input_weight_monitor, fig_weights)
    if j % save_connections_interval == 0 and j > 0 and not test_mode:
        ending = '_' + str(j)
        save_connections(ending)
        save_theta(ending)

    current_spike_count = spikes_ei.num_spikes - previous_spike_count
    previous_spike_count = spikes_ei.num_spikes

    if current_spike_count < 5:
        input_intensity += 1
        group_in.rates = 0 * b.Hz
        b.run(resting_time)
    else:
        result_monitor[j%update_interval,:] = spikes_ei.count[:] - previous_spikes[:]
        if test_mode and use_testing_set:
            input_numbers[j] = testing['y'][j%10000][0]
        else:
            input_numbers[j] = training['y'][j%60000][0]
        # outputNumbers[j,:] = get_recognized_number_ranking(assignments, result_monitor[j%update_interval,:])
        if j % 1000 == 0 and j > 0:
            print('runs done', j, 'of', str(int(num_examples)))
        #if j % update_interval == 0 and j > 0: # performance plotting not supported in this version
            #if do_plot_performance:
            #    unused, performance = update_performance_plot(performance_monitor, performance, j, fig_performance)
            #    print('Classification performance', performance[:(j/float(update_interval))+1])
        group_in.rates = 0 * b.Hz
        b.run(resting_time)
        input_intensity = start_input_intensity
        previous_spikes = spikes_ei.count[:]
        j += 1

#------------------------------------------------------------------------------
# save results
#------------------------------------------------------------------------------
print('save results')
if not test_mode:
    save_theta()
    save_connections()

if test_mode:
    np.save(data_path + 'activity/resultPopVecs' + str(num_examples), result_monitor)
    np.save(data_path + 'activity/inputNumbers' + str(num_examples), input_numbers)

timer_end = datetime.now()
print('End time: ', timer_end)
print('Duration: {}'.format(timer_end - timer_start))
print('exc spikes: ' + str(spikes_ei.num_spikes))
print('inh spikes: ' + str(spikes_ie.num_spikes))

#------------------------------------------------------------------------------
# plot results
#------------------------------------------------------------------------------
# if rate_ei: # plotting for rate monitors does not work
#     b.figure(fig_num)
#     fig_num += 1
#     b.subplot(len(rate_ei), 1, 1)
#     b.plot(rate_ei.t/b.second, rate_ei.rate/b.Hz, '.')
#     b.title('Rates of population Ae')
#     b.subplot(len(rate_ie), 1, 2)
#     b.plot(rate_ie.t/b.second, rate_ie.rate/b.Hz, '.')
#     b.title('Rates of population Ai')

if record_spikes:
    b.figure(fig_num)
    fig_num += 1
    b.subplot(211)
    b.plot(spikes_ei.t/b.ms, spikes_ei.i, '.') # alternatively use "brian_plot(spikes_ei)"
    b.title('Spikes of population Ae')
    b.subplot(212)
    b.plot(spikes_ie.t/b.ms, spikes_ie.i, '.')
    b.title('Spikes of population Ai')

    b.figure(fig_num)
    fig_num += 1
    b.subplot(211)
    b.plot(spikes_ei.count[:])
    b.title('Spike count of population Ae')
    b.subplot(212)
    b.plot(spikes_ie.count[:])
    b.title('Spike count of population Ai')

# plot_2d_input_weights() # input weight plotting not supported in this version
b.ioff()
b.show()




