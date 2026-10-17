// Quantitative values transcribed from Table II of the published paper (p. 11).
window.REMOVAL_RESULTS = {
  showcase: [
    ['austin7_sub2', 'Large roof complex'],
    ['austin28_sub9', 'Parking and roads'],
    ['chicago28_sub10', 'Park and vegetation'],
    ['chicago16_sub5', 'Buildings and greenery'],
    ['innsbruck27_sub4', 'Courtyard and streets'],
    ['tyrol-e6_sub3', 'Roof and grassland'],
    ['vienna12_sub2', 'Large building shadows'],
    ['vienna35_sub1', 'Urban courtyard']
  ],
  comparisons: {
    aisd: ['austin13_sub2', 'chicago34_sub8'],
    asd: ['2012-04-26-Muenchen-Tunnel_4K0G0060_25', '2012-04-26-Muenchen-Tunnel_4K0G0160_82']
  },
  // Column order: PSNR, SSIM, AISD Entropy/BRISQUE/PIQE, ASD Entropy/BRISQUE/PIQE.
  quantitative: [
    {method: 'Chen et al.', domain: '‡', values: ['23.15', '0.881', '2.142', '17.87', '30.98', '2.361', '23.98', '25.45'], std: ['3.96', '0.06']},
    {method: 'Silva et al.', domain: '‡', values: ['23.30', '0.884', '2.151', '19.73', '31.68', '2.362', '22.86', '27.10'], std: ['4.06', '0.05']},
    {method: 'Yu et al.', domain: '‡', values: ['20.61', '0.872', '2.228', '20.19', '26.24', '2.326', '20.34', '25.04'], std: ['3.19', '0.06']},
    {method: 'TBRNet', domain: '†‡', values: ['19.10', '0.867', '2.224', '17.29', '26.75', '2.335', '23.80', '24.16'], std: ['2.59', '0.05']},
    {method: 'ShadowFormer', domain: '†', values: ['22.51', '0.873', '2.222', '17.88', '27.95', '2.345', '23.94', '24.96'], std: ['3.41', '0.06']},
    {method: 'FusionNet', domain: '†', values: ['22.28', '0.852', '2.218', '16.61', '27.10', '2.385', '25.65', '21.26'], std: ['5.09', '0.10']},
    {method: 'DMTN', domain: '†', values: ['24.27', '0.867', '2.300', '20.15', '26.53', '2.405', '21.76', '23.52'], std: ['3.64', '0.05']},
    {method: 'NeFour', domain: '‡', values: ['23.02', '0.876', '2.090', '15.97', '23.47', '2.275', '24.29', '23.57'], std: ['3.97', '0.07']},
    {method: 'SRNet', domain: '‡', values: ['22.71', '0.873', '2.271', '19.75', '22.60', '2.388', '25.70', '20.93'], std: ['3.71', '0.07']},
    {method: 'GSR-Net (Ours)', domain: '', values: ['25.02', '0.887', '2.310', '14.20', '20.97', '2.384', '20.00', '20.91'], std: ['3.08', '0.04']}
  ]
};
