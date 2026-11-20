//! Generic host-controlled policy for diagnostic bodies (guest logs and error details).
use std::fmt::Display;
use std::sync::atomic::{AtomicBool,Ordering};
static CONTENT:AtomicBool=AtomicBool::new(true);
pub fn set_content_logging(enabled:bool){CONTENT.store(enabled,Ordering::Relaxed);}
pub fn detail(value:impl Display)->String{
    if CONTENT.load(Ordering::Relaxed){value.to_string()}else{"[redacted]".into()}
}
#[cfg(test)]
mod tests{
    #[test]
    fn host_can_disable_guest_and_error_bodies(){
        super::set_content_logging(false);
        assert_eq!(super::detail("PRIVATE_CANARY"),"[redacted]");
        super::set_content_logging(true);
    }
}
