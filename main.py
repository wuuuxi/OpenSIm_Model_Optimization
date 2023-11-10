import matplotlib.pyplot as plt
import numpy as np
import opensim as osim
import pandas as pd
import ctypes
import scipy
import torch
import csv
# import net
import os


def change_file_extension(filename, new_extension):
    # Check if the file exists before renaming
    if os.path.isfile(filename):
        file_name, old_extension = os.path.splitext(filename)
        new_filename = file_name + '.' + new_extension
        try:
            os.rename(filename, new_filename)
            print(f"File '{filename}' renamed to '{new_filename}' with new extension.")
        except Exception as e:
            print(f"Error occurred while renaming the file: {e}")
    else:
        print(f"File '{filename}' not found.")


def emg_rectification(x, Fs, code=None):
    # Fs 采样频率，在EMG信号中是1000Hz
    # wp 通带截止频率    ws 阻带截止频率
    x_mean = np.mean(x[1:200])
    raw = x - x_mean * np.ones_like(x)
    t = np.arange(0, raw.size / Fs, 1 / Fs)
    EMGFWR = abs(raw)

    # 线性包络 Linear Envelope
    NUMPASSES = 3
    LOWPASSRATE = 6  # 低通滤波4—10Hz得到包络线

    Wn = LOWPASSRATE / (Fs / 2)
    [b, a] = scipy.signal.butter(NUMPASSES, Wn, 'low')
    EMGLE = scipy.signal.filtfilt(b, a, EMGFWR, padtype='odd', padlen=3*(max(len(b), len(a))-1))  # 低通滤波

    # EMGfig = plt.figure()
    # plt.subplot(2, 1, 1)
    # # plt.plot(t, raw)
    # plt.plot(t, EMGLE)
    # plt.ylabel('Filtered EMG Voltage(\muV)')
    # # plt.show()

    # normalized_EMG = normalize(EMGLE, 'range');
    # ref = max(EMGLE)
    # # yuetian
    # if code == 'BIC':
    #     # ref = 0.403
    #     # ref = 0.55
    #     ref = 0.65
    # elif code == 'BRA':
    #     # ref = 0.273
    #     ref = 0.35
    # elif code == 'BRD':
    #     # ref = 0.235
    #     ref = 0.35
    # else:
    #     ref = max(EMGLE)
    # chenzui
    if code == 'BIC':
        # ref = 0.403
        # ref = 0.2
        ref = 0.65
    elif code == 'BRA':
        # ref = 0.273
        # ref = 0.30
        ref = 0.26
    elif code == 'BRD':
        # ref = 0.235
        # ref = 0.50
        ref = 0.8
    else:
        ref = max(EMGLE)
    # # kehan
    # if code == 'BIC':
    #     # ref = 0.44
    #     ref = 0.52
    # elif code == 'BRA':
    #     # ref = 0.066
    #     ref = 0.19
    # elif code == 'BRD':
    #     # ref = 0.081
    #     ref = 0.15
    #     # ref = 0.25
    # else:
    #     ref = max(EMGLE)
    # if code == 'BIC':
    #     # ref = 3000
    #     # ref = 500
    #     ref = 1000
    # elif code == 'BRA':
    #     ref = 500
    #     # ref = 400
    #     # ref = 1000
    # elif code == 'BRD':
    #     # ref = 1000
    #     ref = 600
    # else:
    #     ref = max(EMGLE)
    # if code == 'BIC':
    #     ref = 1.4
    # elif code == 'BRA':
    #     ref = 0.4
    # elif code == 'BRD':
    #     ref = 0.4
    # else:
    #     ref = max(EMGLE) * 1.4
    # ref = max(EMGLE)
    normalized_EMG = EMGLE / ref
    # # plt.subplot(2, 1, 2)
    # plt.plot(t, normalized_EMG)
    # plt.xlabel('Time(s)')
    # plt.ylabel('Normalized EMG')
    # # plt.ylim(0, 1)
    # # plt.title(code)

    # print(EMGfig, '-dpng', [code '.png'])
    y = normalized_EMG
    # y = EMGLE
    return [y, t]


def resample_by_len(orig_list: list, target_len: int):
    '''
    同于标准重采样，此函数将len(list1)=x 从采样为len(list2)=y；y为指定的值，list2为输出
    :param orig_list: 是list,重采样的源列表：list1
    :param target_len: 重采样的帧数：y
    :return: 重采样后的数组:list2
    '''
    orig_list_len = len(orig_list)
    k = target_len / orig_list_len
    x = [x * k for x in range(orig_list_len)]
    x[-1] = 3572740
    if x[-1] != target_len:
        # 线性更改越界结尾
        x1 = x[-2]
        y1 = orig_list[-2]
        x2 = x[-1]
        y2 = orig_list[-1]
        y_resa = (y2 - y1) * (target_len - x1) / (x2 - x1) + y1
        x[-1] = target_len
        orig_list[-1] = y_resa
    # 使用了线性的插值方法，也可以根据需要改成其他的。推荐是线性
    f = scipy.interpolate.interp1d(x, orig_list, 'linear')
    del x
    resample_list = f([x for x in range(target_len)])
    return resample_list


def find_nearest_idx(arr, value):
    arr = np.asarray(arr)
    array = abs(np.asarray(arr) - value)
    idx = array.argmin()
    return idx


def from_csv_to_muscle_data(csv):
    arr = []
    va = csv.values[6:]
    for i in range(va.size):
        arr.append(va[i, 0].split('\t'))
    arr = np.asarray(arr)

    da = {arr[0, i]: arr[1:, i] for i in range(arr[0].size)}
    dc = pd.DataFrame(da)
    time_csv = [float(i) for i in dc['time']]
    a = dc['bic_s_l']
    b = dc['brachialis_1_l']
    c = dc['brachiorad_1_l']
    batch_csv = [[float(a[i]), float(b[i]), float(c[i])] for i in range(a.size)]
    return batch_csv, time_csv


if __name__ == '__main__':
    # time parameters
    time_start = 3.533
    time_stop = 4.749

    # initial model
    model_init = osim.Model("whole body model/BUET_yuetian_scaled_double_force.osim")

    # print max isometric force of all muscles
    # for muscle in model.getMuscleList():
    #     assert (isinstance(muscle, osim.Thelen2003Muscle))
    #     # assert (isinstance(muscle, osim.Thelen2003Muscle) or
    #     #         isinstance(muscle, osim.Millard2012EquilibriumMuscle))
    #     print(muscle.getName(), muscle.get_max_isometric_force())

    # muscle initial value
    muscle = model_init.getComponent('forceset/bic_s_l')
    assert muscle.getName() == 'bic_s_l'
    print(muscle.getName(), ':', muscle.get_max_isometric_force())  # Method on Muscle.
    muscle = model_init.getComponent('forceset/brachialis_1_l')
    assert muscle.getName() == 'brachialis_1_l'
    print(muscle.getName(), ':', muscle.get_max_isometric_force())  # Method on Muscle.
    muscle = model_init.getComponent('forceset/brachiorad_1_l')
    assert muscle.getName() == 'brachiorad_1_l'
    print(muscle.getName(), ':', muscle.get_max_isometric_force())  # Method on Muscle.
    # muscle = model.updComponent('forceset/LD_Il_l')
    # muscle.set_max_isometric_force(100)

    # initialization of neural network
    # input = [1, 1, 1]
    mif_init = np.asarray([346.0, 628.0, 60.0])
    input = [48.9, 63.0, 8.9, 4.6]  # Peak Torque: extension / flexion of elbow, supination / pronation of forearm
    # mynet = net.TestNet(input_shape=4, output_shape=3, hidden_sizes=[64, 64])
    # loss, error = mynet.init(input=input, output=mif_init, iter=380)
    # print(mynet.net(torch.tensor([input])))
    # plt.plot(loss)
    # plt.figure()
    # plt.subplot(311)
    # plt.plot(np.asarray(error)[:, :, 0])
    # plt.subplot(312)
    # plt.plot(np.asarray(error)[:, :, 1])
    # plt.subplot(313)
    # plt.plot(np.asarray(error)[:, :, 2])
    # plt.show()

    fs = 1000
    emg = pd.read_excel('whole body model/CHENYuetian_10kg.xlsx')
    time_emg = emg['t']
    emg_muscle = [[], [], []]
    emg_time = [[], [], []]
    [emg_muscle[0], emg_time[0]] = emg_rectification(emg['BIC'], fs, 'BIC')
    [emg_muscle[1], emg_time[1]] = emg_rectification(emg['BRA'], fs, 'BRA')
    [emg_muscle[2], emg_time[2]] = emg_rectification(emg['BRD'], fs, 'BRD')

    mifs = []
    # losses = []
    errors = []
    mif = mif_init
    mifs.append(mif)

    module_so = osim.AnalyzeTool("whole body model/Setup_StaticOptimization.xml")
    module_so.setName('whole body_yuetian')
    module_so.setModelFilename("whole body model/BUET_yuetian_scaled_double_force.osim")
    module_so.setExternalLoadsFileName("Arm_Dumbbell_Force_yuetian_scaled_10.xml")
    module_so.setCoordinatesFileName("Chen yuetian_10kg_Xsens_jointangle_q.mot")
    # module_so.setCoordinatesFileName("modelModified_Kinematics_q.mot")
    module_so.setResultsDir("Results")
    module_so.setInitialTime(time_start)
    module_so.setFinalTime(time_stop)

    for ir in range(1000):
        print('-' * 40, 'Cycle NO.', ir, '-' * 40)

        module_so.run()

        change_file_extension('whole body model/Results/whole body_yuetian_StaticOptimization_activation.sto', 'csv')
        df = pd.read_csv('whole body model/Results/whole body_yuetian_StaticOptimization_activation.csv')
        act_cal, time_cal = from_csv_to_muscle_data(df)
        print('emg activation:\t', np.mean(act_cal[0]), np.mean(act_cal[1]), np.mean(act_cal[2]))

        batch_ratio = []
        batch_error = []
        idx = [0, 0, 0]
        for i in range(len(time_cal)):
            idx = np.asarray([find_nearest_idx(emg_time[j], time_cal[i]) for j in range(3)])
            ratio = np.asarray([act_cal[i][j] / emg_muscle[j][idx[j]] for j in range(3)])
            error = np.asarray([act_cal[i][j] - emg_muscle[j][idx[j]] for j in range(3)])
            batch_ratio.append(ratio)
            batch_error.append(error)
        batch_ratio = np.asarray(batch_ratio)
        print('ratio:\t', np.asarray([batch_ratio[:, i].mean() for i in range(3)]))
        mif = np.asarray([batch_ratio[:, i].mean() * mif[i] for i in range(3)])
        mifs.append(mif)
        errors.append(batch_error)

        # for _ in range(5):
        #     loss, error = mynet.update(input, torch.tensor(batch))
        # loss, error = mynet.update(input, torch.tensor(batch))
        # losses.append(loss)
        # errors.append(error)
        # print(mynet.net(torch.tensor([input])))

        module_so = osim.AnalyzeTool("whole body model/Setup_StaticOptimization.xml")
        module_so.setName('whole body_yuetian')
        module_so.setModelFilename("whole body model/BUET_yuetian_scaled_double_force.osim")
        module_so.setExternalLoadsFileName("Arm_Dumbbell_Force_yuetian_scaled_10.xml")
        module_so.setCoordinatesFileName("Chen yuetian_10kg_Xsens_jointangle_q.mot")
        # module_so.setCoordinatesFileName("modelModified_Kinematics_q.mot")
        module_so.setResultsDir("Results")
        module_so.setInitialTime(time_start)
        module_so.setFinalTime(time_stop)

        model = module_so.getModel()
        muscle = model.updComponent('forceset/bic_s_l')
        muscle.set_max_isometric_force(float(mif[0]))
        muscle = model.updComponent('forceset/brachialis_1_l')
        muscle.set_max_isometric_force(float(mif[1]))
        muscle = model.updComponent('forceset/brachiorad_1_l')
        muscle.set_max_isometric_force(float(mif[2]))

        model = module_so.getModel()
        muscle = model.getComponent('forceset/bic_s_l')
        assert muscle.getName() == 'bic_s_l'
        print(muscle.getName(), ':\t', muscle.get_max_isometric_force())
        muscle = model.getComponent('forceset/brachialis_1_l')
        assert muscle.getName() == 'brachialis_1_l'
        print(muscle.getName(), ':\t', muscle.get_max_isometric_force())
        muscle = model.getComponent('forceset/brachiorad_1_l')
        assert muscle.getName() == 'brachiorad_1_l'
        print(muscle.getName(), ':\t', muscle.get_max_isometric_force())

        np.save('max_isometric_force.npy', mifs)
        # np.save('loss.npy', losses)
        np.save('error.npy', errors)

    np.save('max_isometric_force.npy', mifs)
    # np.save('loss.npy', losses)
    np.save('error.npy', errors)
    plt.figure()
    plt.plot(np.asarray(mifs)[:, 0])
    # plt.figure()
    # plt.plot(np.asarray(losses)[:, 0])
    plt.figure()
    e = np.asarray(errors)
    u = e.reshape(e.shape[0]*e.shape[1], e.shape[2])
    plt.plot(u[:, 0])
    plt.show()
    # change_file_extension('whole body model/whole body_yuetian_StaticOptimization_activation.sto', 'csv')
    # df = pd.read_csv('whole body model/whole body_yuetian_StaticOptimization_activation.csv')
    # data_storage = osim.Storage("/whole body model/whole body_yuetian_StaticOptimization_activation.sto")

    # model = osim.Model("whole body model/BUET_yuetian_scaled_double_force.osim")
    # model = osim.Model("../model/MOBL/MOBL_ARMS_YUETIAN.osim")  # model
    # data_storage = osim.Storage("../model/MOBL/MOBL_Motion_yuetian_10.mot")  # kinematics
    #  still need: time range, residual actuators, external loads

    # muscle = model.getComponent('forceset/BICshort')
    # assert muscle.getName() == 'BICshort'
    # muscle.get_max_isometric_force()  # Method on Muscle.
    # muscle = model.updComponent('forceset/BICshort')
    # muscle.set_max_isometric_force(100)

    # muscle = model.getComponent('forceset/LD_Il_l')
    # assert muscle.getName() == 'LD_Il_l'
    # muscle.get_max_isometric_force()  # Method on Muscle.
    # muscle = model.updComponent('forceset/LD_Il_l')
    # muscle.set_max_isometric_force(100)

    # trials = osim.Storage("../model/MOBL/MOBL_Motion_yuetian_10.mot")  # kinematics
    # state = model.initSystem()
    # model.equilibrateMuscles(state)

    # ana = osim.AnalyzeTool("subject01_Setup_StaticOptimization.xml")
    # ana = osim.AnalyzeTool("../model/whole body model/Setup_StaticOptimization.xml")
    # ana.setModel()
    # ana.run()

    # b = osim.StaticOptimization()
    # b.getStartTime()
    # b.getMaxIterations()
    # b.getConvergenceCriterion()
    # b.setConvergenceCriterion()
    # b.setMaxIterations()
    # b.setStartTime(time_start)
    # b.setEndTime(time_stop)
    # b.setStepInterval(10)
    # b.setInDegrees(True)
    # b.setUseModelForceSet(True)
    # b.setActivationExponent(2)
    # b.setUseMusclePhysiology(True)

    # model = osim.Model("../model/whole body model/BUET_yuetian_scaled_double_force.osim")
    # data_storage = osim.Storage("../model/whole body model/modelModified_Kinematics_q.mot")
    # data_storage = osim.Storage("/whole body model/whole body_yuetian_StaticOptimization_activation.sto")
    # data_storage.getDataAtTime()
    # data_storage.getStateIndex('bic_s_l')
    # data_storage.getStateIndex('brachialis_1_l')
    # data_storage.getStateVector()
    # data_storage.getColumnLabels().size()
    # data_storage.getSize()
    # data_storage.getDataColumn()
    # data_storage.getColumnLabels().get(5)
    # data_storage.getStateVector(5).getTime()
    # # data_storage.getDataColumn().get(5)
    # data_storage.getStateIndex('pelvis_ty')
    # data_storage.compareColumn(data_storage, 'pelvis_ty', 0, 1)
    # data_storage.exportToTable()
    # data_storage.getStateVector(1).getDataValue(1, )

    # model.updForceSet()
    # b.setModel(model)
    # b.setModel("../model/whole body model/BUET_yuetian_scaled_double_force.osim")

    # a = osim.AnalyzeTool("/whole body model/Setup_StaticOptimization.xml")
    # a.setName('whole body_yuetian')
    # a.setModelFilename("/whole body model/BUET_yuetian_scaled_double_force.osim")
    # a.setExternalLoadsFileName("Arm_Dumbbell_Force_yuetian_scaled_10.xml")
    # # a.setExternalLoadsFileName("../model/whole body model/Arm_Dumbbell_Force_yuetian_scaled_10.xml")
    # # a.setStatesStorage(data_storage)
    # a.setCoordinatesFileName("Chen yuetian_10kg_Xsens_jointangle_q.mot")
    # a.setCoordinatesFileName("modelModified_Kinematics_q.mot")
    # a.setResultsDir("Results_yuetian_StaticOptimization")
    # a.setInitialTime(time_start)
    # a.setFinalTime(time_stop)
    # # a.addAnalysisSetToModel()
    # # a.updAnalysisSet()
    # a.getAnalysisSet()
    # # a.setModel()
    # a.run()


    # model = osim.Model("../model/whole body model/BUET_yuetian_scaled_double_force.osim")
    # model.addAnalysis(b)
    # a.setModel(model)
    # data_storage = osim.Storage("../model/whole body model/Chen yuetian_10kg_Xsens_jointangle_q.mot")
    # a.setCoordinatesFileName("../model/whole body model/Chen yuetian_10kg_Xsens_jointangle_q.mot")


    # model = osim.Model("../model/whole body model/BUET_yuetian_scaled_double_force.osim")
    # model.addAnalysis(b)
    # data_storage = osim.Storag("../model/whole body model/Chen yuetian_10kg_Xsens_jointangle_q.mot")
    # a.setModel(model)
    # c = a.updAnalysisSet()
    # # print(c)
    # c = osim.analyze(b)
    # c = osim.analyzeVec3(b)
    # print(c)
    # a.updAnalysisSet()
    # b.step()

    # net = Net(state_shape=3, action_shape=3, hidden_sizes=[64, 64], activation=nn.Tanh)





    # # numControls = model.getNumControls()  # The number of controls will equal the number of muscles in the model
    # # print(numControls)
    # SO = osim.StaticOptimization(model)
    # # SO.setUseModelForcubuntueSet()
    # # SO.setModel(model)
    # SO.setStartTime(0.0)
    # SO.setEndTime(0.1)
    # # SO.setStorageCapacityIncrements(trials)
    # SO.begin(state)
    # SO.record
    # a = SO.step(state, 2)
    # print(a)
    # SO.setUseMusclePhysiology()
    # SO.setUseModelForceSet()

    # for muscle in model.getMuscleList():
    #     assert (isinstance(muscle, osim.Thelen2003Muscle))
    #     # assert (isinstance(muscle, osim.Thelen2003Muscle) or
    #     #         isinstance(muscle, osim.Millard2012EquilibriumMuscle))
    #     print(muscle.getName(), muscle.get_max_isometric_force())

    # muscle = osim.Thelen2003Muscle()
    # property = muscle.getPropertyByName('max_isometric_force')
    # osim.PropertyHelper.setValueDouble(200, property)
    # # assert muscle.get_max_isometric_force() == 200
    # print(muscle.get_max_isometric_force())



    # for force in model.getForceSet():
    #     print(force.getName())
    # a = model.getMuscles()
    # print(a)
    # a = model.getMuscles()
    # data_storage
    # a = model.getMuscleList().set_max_isometric_force()
    # a = model.getMuscles().set_max_isometric_force()
    # print(a)
    # model.set_max_isometric_force()
    # model.setMaxIsometricForce()
    # for muscle in model.getMuscles():
    #     print(muscle.get_max_isometric_force())
    # for name, muscle in model.getMuscles():
    #     print(name, muscle.get_max_isometric_force())
    # for body in model.getBodyList():
    #     print(body.getName())

