const fs = require('fs');
const glob = require('glob');

const paths = glob.sync('src/**/*.js');

paths.forEach(file => {
  let code = fs.readFileSync(file, 'utf8');
  let changed = false;

  // Replace obj?.arr?.map(...)
  const newCode = code.replace(/(\w+(?:\?\.\w+)+)\?\.map\(/g, '(Array.isArray($1) ? $1 : []).map(')
                      .replace(/(\w+(?:\?\.\w+)*)\.map\(/g, (match, p1) => {
                         if(p1 === 'Array' || p1 === 'Promise') return match;
                         // Just avoiding basic ones
                         // Actually, this regex is too broad to just replace everything. 
                         return match;
                      });

  if (newCode !== code) {
    //fs.writeFileSync(file, newCode, 'utf8');
  }
});
