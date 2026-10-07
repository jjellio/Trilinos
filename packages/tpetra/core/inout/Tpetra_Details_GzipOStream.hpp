#ifndef TPETRA_DETAILS_GZIPOSTREAM_HPP
#define TPETRA_DETAILS_GZIPOSTREAM_HPP

#include <memory>
#include <ostream>
#include <string>

namespace Tpetra {
namespace Details {

// Open filename for Matrix Market output.
//
//   *.gz  -> gzip-compressed output when TpetraCore has Zlib enabled
//   other -> ordinary std::ofstream
//
// The implementation lives in Tpetra_Details_GzipOStream.cpp so that ETI
// translation units do not compile the zlib stream-buffer implementation.
std::unique_ptr<std::ostream>
openMatrixMarketOutputStream(const std::string& filename,
    std::ios_base::openmode mode = std::ios_base::out);

} // namespace Details
} // namespace Tpetra

#endif // TPETRA_DETAILS_GZIPOSTREAM_HPP

