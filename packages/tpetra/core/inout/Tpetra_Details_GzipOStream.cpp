#include "Tpetra_Details_GzipOStream.hpp"
#include "TpetraCore_config.h"

#include <fstream>
#include <stdexcept>

#ifdef HAVE_TPETRACORE_ZLIB
#include <zlib.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <streambuf>
#endif

namespace {

bool
hasGzipSuffix(const std::string& filename)
{
  static const std::string suffix = ".gz";
  return filename.size() >= suffix.size() &&
         filename.compare(filename.size() - suffix.size(),
                          suffix.size(), suffix) == 0;
}

#ifdef HAVE_TPETRACORE_ZLIB

class GzipStreamBuf : public std::streambuf {
public:
  GzipStreamBuf(const std::string& filename, const char* gzipMode)
    : file_(gzopen(filename.c_str(), gzipMode))
  {
    if (file_ == nullptr) {
      throw std::runtime_error(
          "Failed to open gzip output file \"" + filename + "\"");
    }
    setp(buffer_.data(), buffer_.data() + buffer_.size());
  }

  ~GzipStreamBuf() override
  {
    closeNoThrow();
  }

  GzipStreamBuf(const GzipStreamBuf&) = delete;
  GzipStreamBuf& operator=(const GzipStreamBuf&) = delete;

  bool close()
  {
    if (file_ == nullptr) {
      return !failed_;
    }

    bool ok = flushBuffer();
    const int status = gzclose(file_);
    file_ = nullptr;

    if (status != Z_OK) {
      ok = false;
    }

    failed_ = failed_ || !ok;
    return !failed_;
  }

protected:
  int_type overflow(int_type ch) override
  {
    if (traits_type::eq_int_type(ch, traits_type::eof())) {
      if (flushBuffer()) {
        return traits_type::not_eof(ch);
      }
      return traits_type::eof();
    }

    if (!flushBuffer()) {
      return traits_type::eof();
    }

    *pptr() = traits_type::to_char_type(ch);
    pbump(1);
    return ch;
  }

  std::streamsize xsputn(const char* data, std::streamsize count) override
  {
    std::streamsize total = 0;

    while (total < count) {
      const std::streamsize available = epptr() - pptr();

      if (available == 0) {
        if (!flushBuffer()) {
          break;
        }
        continue;
      }

      const std::streamsize remaining = count - total;
      const std::streamsize amount = remaining < available ? remaining : available;

      std::memcpy(pptr(), data + total, static_cast<std::size_t>(amount));
      pbump(static_cast<int>(amount));
      total += amount;
    }

    return total;
  }

  int sync() override
  {
    return flushBuffer() ? 0 : -1;
  }

private:
  static constexpr std::size_t bufferSize_ = 64 * 1024;

  bool flushBuffer()
  {
    if (failed_ || file_ == nullptr) {
      return false;
    }

    const std::ptrdiff_t count = pptr() - pbase();
    if (count == 0) {
      return true;
    }

    const unsigned int amount = static_cast<unsigned int>(count);
    const int written = gzwrite(file_, pbase(), amount);

    if (written != static_cast<int>(amount)) {
      failed_ = true;
      return false;
    }

    setp(buffer_.data(), buffer_.data() + buffer_.size());
    return true;
  }

  void closeNoThrow() noexcept
  {
    if (file_ == nullptr) {
      return;
    }

    (void) flushBuffer();
    if (gzclose(file_) != Z_OK) {
      failed_ = true;
    }
    file_ = nullptr;
  }

  gzFile file_ = nullptr;
  bool failed_ = false;
  std::array<char, bufferSize_> buffer_{};
};

class GzipOStream : public std::ostream {
public:
  GzipOStream(const std::string& filename, const char* gzipMode)
    : std::ostream(nullptr), buffer_(filename, gzipMode)
  {
    rdbuf(&buffer_);
    clear();
  }

  ~GzipOStream() override
  {
    (void) close();
  }

  GzipOStream(const GzipOStream&) = delete;
  GzipOStream& operator=(const GzipOStream&) = delete;

  bool close() noexcept
  {
    if (closed_) {
      return !bad();
    }

    flush();
    bool ok = !bad();
    ok = buffer_.close() && ok;

    closed_ = true;
    if (!ok) {
      setstate(std::ios::badbit);
    }
    return ok;
  }

private:
  GzipStreamBuf buffer_;
  bool closed_ = false;
};

#endif // HAVE_TPETRACORE_ZLIB

} // namespace

namespace Tpetra {
namespace Details {

std::unique_ptr<std::ostream>
openMatrixMarketOutputStream(
    const std::string& filename,
    const std::ios_base::openmode mode)
{
  if (hasGzipSuffix(filename)) {
#ifdef HAVE_TPETRACORE_ZLIB
    const std::ios_base::openmode zero =
        static_cast<std::ios_base::openmode>(0);

    const bool append = (mode & std::ios_base::app) != zero;
    const bool atEnd  = (mode & std::ios_base::ate) != zero;
    const bool input  = (mode & std::ios_base::in) != zero;

    if (input || atEnd) {
      throw std::runtime_error(
          "Unsupported open mode for gzip Matrix Market output file \"" +
          filename + "\"");
    }

    return std::unique_ptr<std::ostream>(
        new GzipOStream(filename, append ? "ab" : "wb"));
#else
    throw std::runtime_error(
        "Cannot write gzip-compressed Matrix Market file \"" + filename +
        "\": TpetraCore was built without Zlib support.");
#endif
  }

  std::unique_ptr<std::ofstream> out(
      new std::ofstream(filename.c_str(), mode));
  if (!(*out)) {
    throw std::runtime_error(
        "Failed to open Matrix Market output file \"" + filename + "\"");
  }

  return std::unique_ptr<std::ostream>(out.release());
}

} // namespace Details
} // namespace Tpetra

