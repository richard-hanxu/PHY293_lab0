import math
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

def option1():
    I = [62.83, 28.933, 13.769, 2.368]
    I = [i * 10**-3 for i in I]
    I_unc = [.005, .0005, .0005, .0005]
    I_unc = [i * 10**-3 for i in I_unc]

    Rl = [99.85, 218.72, 462.2, 2698.9]
    Rl_unc = [.005, .005, .05, .05]

    V = [6.376, 6.384, 6.392, 6.395]
    V_unc = 0.0005

    R_a = [V / current - load_resistance for V, current, load_resistance in zip(V, I, Rl)]
    R_aunc = [math.sqrt((V_unc / current)**2 + (V * I_unc / current**2)**2 + (load_resistance_unc)**2) for V, current, I_unc, load_resistance_unc in zip(V, I, I_unc, Rl_unc)]
    unc_avg = math.sqrt(sum([unc**2 for unc in R_aunc])/len(R_aunc))
    #print(f'{(V_unc / I[3])**2}, {(V[3] * I_unc[3] / I[3]**2)**2}', {(Rl_unc[3])**2})
    #print(R_aunc[3])
    with open('option1.txt', 'w') as f:
        f.write('R_a,R_aunc,V,V_unc,I,I_unc,Rl,Rl_unc\n')
        for i in range(len(R_a)):
            f.write(f'{R_a[i]},{R_aunc[i]},{V[i]},{V_unc},{I[i]},{I_unc},{Rl[i]},{Rl_unc[i]}\n')
        f.write(f'Average Resistance: {sum(R_a)/len(R_a)}\nAverage Uncertainty: {unc_avg}')
        f.close()

    return I, V, R_a, R_aunc

def option2():
    I = [62.61, 28.920, 13.760, 2.367]
    #I = [62.83, 29.000, 23.798, 2.373]
    I = [i * 10**-3 for i in I] 
    I_unc = [.005, .0005, .0005, .0005]
    I_unc = [i * 10**-3 for i in I_unc]

    Rl = [99.85, 218.72, 462.2, 2698.9]
    Rl_unc = [.005, .005, .05, .05]

    V = [6.261, 6.329, 6.364, 6.388]
    #V = [6.277, 6.347, 6.379, 6.402]
    V_unc = 0.0005

    R_v = [Rl * V / (V - current * Rl) for V, current, Rl in zip(V, I, Rl)]
    dRv_dV = [(V_unc * curr / (curr - V/Rl)**2)**2 for V, curr, Rl in zip(V, I, Rl)]
    dRv_dI = [(I_unc * V / (curr - V/Rl)**2)**2 for V, curr, I_unc, Rl in zip(V, I, I_unc, Rl)]
    dRv_dRl = [(Rl_unc * (V/Rl)**2 / (curr - V/Rl)**2)**2 for V, curr, Rl, Rl_unc in zip(V, I, Rl, Rl_unc)]
    R_vunc = [math.sqrt(unc_i + unc_r + unc_v) for unc_i, unc_r, unc_v in zip(dRv_dI, dRv_dRl, dRv_dV)]
    unc_avg = math.sqrt(sum([unc**2 for unc in R_vunc])/len(R_vunc))

    with open('option2.txt', 'w') as f:
        f.write('R_v,R_vunc,V,V_unc,I,I_unc,Rl,Rl_unc\n')
        for i in range(len(R_v)):
            f.write(f'{R_v[i]},{R_vunc[i]},{V},{V_unc},{I[i]},{I_unc[i]},{Rl[i]},{Rl_unc[i]}\n')
        f.write(f'Average Resistance: {sum(R_v)/len(R_v)}\nAverage Uncertainty: {unc_avg}')
        f.close()

    return I, V, R_v, R_vunc 

def residuals(observed, expected):
    return [(observed[i] - expected[i]) for i in range(len(observed))]

def chi_squared(observed, expected, uncertainty):
    return sum([(observed[i] - expected[i])**2 / uncertainty[i]**2 for i in range(len(observed))])

# Returns plots of V vs I and Resistance vs Resistance Uncertainty
def plot_option(curr, volt, s):
    plt.figure(1)
    curr = [i * 10**3 for i in curr]

    # Get uncertainty of slope, y-intercept
    print(curr)
    print(volt)
    slope, intercept, r_value, p_value, std_err = stats.linregress(curr, volt)
    confidence_interval = 2.58*std_err
    print(f'Slope: {slope}, Intercept: {intercept}, Confidence Interval: {confidence_interval}')

    # Plot line of best fit
    plt.clf()
    plt.errorbar(curr, volt, yerr=0.0005, fmt='o', label='Measured Voltage')
    plt.plot(curr, slope * np.array(curr) + intercept, label='Linear Fit')
    plt.annotate(f"Line of best fit: y = {slope:.4e}x + {intercept:.4e}", xy=(0.05, 0.90), xycoords='axes fraction', label="Expected Voltage")
    plt.xlabel("Current (mA)")
    plt.ylabel("Voltage (V)")
    plt.title("Voltage vs Current")
    plt.legend(loc='best')
    #plt.show()
    plt.savefig(f'{s}_linear.png', bbox_inches='tight')

    # Chi Squared
    expected = [slope * i + intercept for i in curr]
    chi = chi_squared(volt, expected, [0.0005 for i in volt])
    residual = residuals(volt, expected)
    print(f'Chi Squared: {chi}')

    # Plot residuals graph
    plt.clf()
    plt.errorbar(curr, residual, yerr=0.0005, fmt='o', label="Measured Voltage")
    plt.plot((0, max(curr)), (0, 0), label="Expected Voltage")
    plt.xlabel("Current (mA)")
    plt.ylabel("Voltage Residual (V)")
    plt.title("Voltage Residual vs Current")
    plt.legend(loc='best')
    #plt.show()
    plt.savefig(f'{s}_residual.png', bbox_inches='tight')

    return slope, confidence_interval

def calc_r(resistance_v, unc_v, resistance_a, unc_a, m_1, m_2, m_1_unc, m_2_unc):
    m_1 *= 10**3 # Scaling units so that it becomes V/A
    m_1_unc *= 10**3
    zipped_v = zip(resistance_v, unc_v)
    filter_v = list(filter(lambda val: val[0] > 0, zipped_v))
    unzipped_v = list(zip(*filter_v))
    resistance_v = list(unzipped_v[0])
    unc_v = list(unzipped_v[1])
    average_v = sum(resistance_v) / len(resistance_v)
    r1 = average_v * m_1/ (m_1 - average_v)
    average_v_unc = math.sqrt(sum([unc**2 for unc in unc_v]))/len(unc_v)
    print(average_v)
    print(f'Average Voltage: {average_v}, Uncertainty of Average: {average_v_unc}')
    r1_unc = math.sqrt((average_v_unc / ((m_1 * average_v) - 1))**2 + ((m_1_unc * average_v) / ((m_1 * average_v) - 1)**2)**2)
    
    m_2 *= 10**3 # Scaling units so that it becomes V/A
    m_2_unc *= 10**3
    zipped_a = zip(resistance_a, unc_a)
    filter_a = list(filter(lambda val: val[0] > 0, zipped_a))
    unzipped_a = list(zip(*filter_a))
    resistance_a = list(unzipped_a[0])
    unc_a = list(unzipped_a[1])
    average_a = sum(resistance_a) / len(resistance_a)
    r2 = -1 * m_2 - average_a
    average_a_unc = math.sqrt(sum([unc**2 for unc in unc_a]))/len(unc_a)
    print(f'Average Current: {average_a}, Uncertainty of Average: {average_a_unc}')
    r2_unc = math.sqrt(average_a_unc**2 + m_2_unc**2)

    return r1, r1_unc, r2, r2_unc

# Run option1 and option2 to get the Rv and Ra values and uncertainties
I1, V1, R_a, R_aunc = option1()
I2, V2, R_v, R_vunc = option2()

# Run plot option to get the plotted lines of best fit and residuals
# Chi squared is also calculated and printed out
m1, m1_unc = plot_option(I1, V1, "option1")
m2, m2_unc = plot_option(I1, V2, "option2")

# Calculate R1 and R2 and their uncertainties
r1, r1_unc, r2, r2_unc = calc_r(R_v, R_vunc, R_a, R_aunc, m1, m2, m1_unc, m2_unc)

print(f'R1: {r1} +/- {r1_unc}')
print(f'R2: {r2} +/- {r2_unc}')