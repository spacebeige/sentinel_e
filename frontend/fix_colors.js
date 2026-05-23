const fs = require('fs');
const path = require('path');

function walkDir(dir, callback) {
  fs.readdirSync(dir).forEach(f => {
    let dirPath = path.join(dir, f);
    let isDirectory = fs.statSync(dirPath).isDirectory();
    isDirectory ? walkDir(dirPath, callback) : callback(path.join(dir, f));
  });
}

walkDir('src', function(filePath) {
  if (filePath.endsWith('.js') || filePath.endsWith('.css')) {
    let content = fs.readFileSync(filePath, 'utf8');
    let original = content;

    // Replace colors
    content = content.replace(/#3b82f6/g, '#1c1c1e');
    content = content.replace(/#06b6d4/g, '#6e6e73');
    
    // Replace specific tailwind gradients
    content = content.replace(/to-cyan-500/g, 'to-gray-500');
    content = content.replace(/from-emerald-400/g, 'from-gray-400');
    content = content.replace(/from-blue/g, 'from-gray');
    content = content.replace(/shadow-cyan/g, 'shadow-gray');
    content = content.replace(/bg-blue/g, 'bg-gray');

    if (content !== original) {
      fs.writeFileSync(filePath, content, 'utf8');
      console.log('Updated', filePath);
    }
  }
});
