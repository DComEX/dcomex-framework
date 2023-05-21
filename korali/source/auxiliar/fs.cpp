#include "auxiliar/fs.hpp"
#include "string.h"
#include <dirent.h>
#include <string>
#include <sys/stat.h>

namespace korali
{
void mkdir(const std::string dirPath)
{
  char tmp[256];
  char *p = NULL;
  size_t len;

  snprintf(tmp, sizeof(tmp), "%s", dirPath.c_str());
  len = strlen(tmp);
  if (tmp[len - 1] == '/')
    tmp[len - 1] = 0;
  for (p = tmp + 1; *p; p++)
    if (*p == '/')
    {
      *p = 0;
      ::mkdir(tmp, S_IRWXU);
      *p = '/';
    }
  ::mkdir(tmp, S_IRWXU);
}


bool dirExists(const std::string dirPath)
{
  DIR *dir = opendir(dirPath.c_str());
  if (dir)
  {
    closedir(dir);
    return true;
  }

  return false;
}

} // namespace korali
