use std::io;
use std::num;
use std::string::FromUtf8Error;

error_chain! {
    foreign_links {
        Io(io::Error);
        ParseInt(num::ParseIntError);
        ParseFloat(num::ParseFloatError);
        Decoding(FromUtf8Error);
    }
}
