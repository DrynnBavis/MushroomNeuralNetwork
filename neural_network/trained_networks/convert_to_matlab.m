clear
files = dir('*.mat')

for file = files'
  load(file.name);
  save('-mat7-binary', file.name);
end
